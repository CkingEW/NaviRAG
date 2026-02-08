import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import re
import time
import hashlib
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Assuming these imports exist in the user's environment structure
from navirag.llm import _get_llm_class, BaseLLM
from navirag.embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from navirag.embedding_store import EmbeddingStore
from navirag.information_extraction import OpenIE
from navirag.information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from navirag.evaluation.retrieval_eval import RetrievalRecall
from navirag.evaluation.qa_eval import QAExactMatch, QAF1Score
from navirag.prompts.linking import get_query_instruction
from navirag.prompts.prompt_template_manager import PromptTemplateManager
from navirag.utils.misc_utils import *
from navirag.utils.misc_utils import NerRawOutput, TripleRawOutput
from navirag.utils.embed_utils import retrieve_knn
from navirag.utils.typing import Triple
from navirag.utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)

# ==============================================================================
#  Helper Classes for RL
# ==============================================================================

class PolicyNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        """
        Initializes the policy network with an LSTM for history and MLPs for state and action scoring.
        """
        super(PolicyNetwork, self).__init__()
        self.history_rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.state_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, current_node_emb, query_emb, history_embs, neighbor_embs):
        """
        Forward pass to compute action probabilities for neighbor nodes based on the current state.
        """
        _, (h_n, _) = self.history_rnn(history_embs)
        history_context = h_n.squeeze(0)
        state_input = torch.cat([current_node_emb, query_emb, history_context], dim=1)
        state_context = self.state_mlp(state_input)
        if neighbor_embs.shape[0] == 0:
            return Categorical(torch.tensor([]))
        state_context_repeated = state_context.repeat(neighbor_embs.shape[0], 1)
        action_input = torch.cat([state_context_repeated, neighbor_embs], dim=1)
        scores = self.action_scorer(action_input).squeeze(1)
        action_probs = F.softmax(scores, dim=-1)
        return Categorical(action_probs)


class RLAgent:
    def __init__(self, policy_network, learning_rate: float, gamma: float):
        """
        Initializes the RL Agent with a policy network, optimizer, and reward discount factor.
        """
        self.device = torch.device("cuda")
        self.policy = policy_network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []
    
    def select_action(self, state_embs):
        """
        Selects an action based on the policy distribution and returns the action index and log probability.
        """
        current_node_emb, query_emb, history_embs, neighbor_embs = state_embs
        if neighbor_embs.shape[0] == 0: return None, None
        
        dist = self.policy(current_node_emb, query_emb, history_embs, neighbor_embs)
        if dist.probs.nelement() == 0: return None, None
        
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        
        self.saved_log_probs.append(log_prob) 
        return action_index.item(), log_prob.item()

    #Reward Shaping
    def run_episode(self, navi_rag_instance, start_node, query, gold_docs, gold_doc_embs, 
                        max_steps, reward_gold, reward_step):
        """
        Runs a target-driven episode where rewards are based on similarity to gold documents (the answer).
        """
        path = [start_node]
        current_node = start_node
        
        # 1. Prepare Query Embedding 
        query_emb_np = navi_rag_instance.query_to_embedding['passage'].get(query, None)
        if query_emb_np is None:
             query_emb_np = navi_rag_instance.embedding_model.batch_encode(query, 
                                                             instruction=get_query_instruction('query_to_passage'), 
                                                             norm=True)
             navi_rag_instance.query_to_embedding['passage'][query] = query_emb_np
        query_emb = torch.from_numpy(query_emb_np).float().view(1, -1).to(self.device)

        # 2. Prepare Start Node Embedding                 
        current_node_emb = navi_rag_instance.get_embedding_for_node(current_node)
        if current_node_emb is not None:
            current_node_emb = current_node_emb.view(1, -1)
        else:
            logger.warning(f"Start node {start_node} has NO embedding! Aborting episode.")
            return path
           
        # Calculate current distance to target (Baseline)
        if gold_doc_embs is not None and len(gold_doc_embs) > 0:
            gold_doc_embs = gold_doc_embs.to(self.device)
            max_sim_to_gold_current = torch.max(F.cosine_similarity(current_node_emb, gold_doc_embs)).item()
        else:
            max_sim_to_gold_current = 0.0

        found_gold = False
        max_sim_reached = max_sim_to_gold_current 

        for step in range(max_steps):
            # --- Action Selection ---
            try:
                neighbor_indices = navi_rag_instance.graph.neighbors(navi_rag_instance.node_name_to_vertex_idx[current_node])
                neighbor_names = [navi_rag_instance.graph.vs[i]["name"] for i in neighbor_indices if navi_rag_instance.graph.vs[i]["name"] not in path]
                if not neighbor_names: break
                
                neighbor_embs_list, valid_neighbor_names = [], []
                for n in neighbor_names:
                    emb = navi_rag_instance.get_embedding_for_node(n)
                    if emb is not None:
                        neighbor_embs_list.append(emb)
                        valid_neighbor_names.append(n)
                
                if not valid_neighbor_names: break
                neighbor_names = valid_neighbor_names
                
                neighbor_embs = torch.stack(neighbor_embs_list) 
                history_embs_list = [navi_rag_instance.get_embedding_for_node(n) for n in path]
                valid_history_embs = [e for e in history_embs_list if e is not None]
                if not valid_history_embs: continue
                history_embs = torch.stack(valid_history_embs).unsqueeze(0)
                
                state_embs = (current_node_emb, query_emb, history_embs, neighbor_embs)
                action_info = self.select_action(state_embs)
                if action_info is None: break
                action_index, log_prob = action_info
                
                next_node = neighbor_names[action_index]
                next_node_emb = neighbor_embs_list[action_index].view(1, -1)
            
            except Exception as e:
                logger.debug(f"run_episode Error: {e}"); break

            # --- Reward Calculation (Target-Driven) ---
            
            # 1. Calculate similarity of new node to target
            if gold_doc_embs is not None and len(gold_doc_embs) > 0:
                max_sim_to_gold_next = torch.max(F.cosine_similarity(next_node_emb, gold_doc_embs)).item()
            else:
                max_sim_to_gold_next = 0.0
            
            # 2. Progress reward
            progress = max_sim_to_gold_next - max_sim_to_gold_current
            progress_reward = max(0, progress) 
            
            # 3. Combined Reward
            reward = reward_step 
            reward += progress_reward * 2.0 
            
            # 4. Gold Reward (Hard Reward)
            if next_node in navi_rag_instance.passage_node_keys:
                passage_content = navi_rag_instance.chunk_embedding_store.get_row(next_node)["content"]
                if passage_content in gold_docs:
                    reward += reward_gold
                    found_gold = True
                    self.rewards.append(reward)
                    path.append(next_node)
                    break 

            self.rewards.append(reward)

            # --- State Transition ---
            path.append(next_node)
            current_node = next_node
            current_node_emb = next_node_emb
            max_sim_to_gold_current = max_sim_to_gold_next
            max_sim_reached = max(max_sim_reached, max_sim_to_gold_next)

        # --- Soft Terminal Reward ---
        if not found_gold and gold_doc_embs is not None and self.rewards:
            if max_sim_reached > 0: 
                soft_reward = reward_gold * max_sim_reached
                self.rewards[-1] += soft_reward

        return path

    def run_episode_query(self, navi_rag_instance, start_node, query, gold_docs, gold_doc_embs, 
                            max_steps, reward_gold, reward_step):
            """
            Runs an episode where rewards are driven by query similarity (Query-Driven Progress). 
            Used primarily for ablation studies.
            """
            path = [start_node]
            current_node = start_node
            
            query_emb_np = navi_rag_instance.query_to_embedding['passage'].get(query, None)
            if query_emb_np is None:
                 query_emb_np = navi_rag_instance.embedding_model.batch_encode(query, 
                                                                 instruction=get_query_instruction('query_to_passage'), 
                                                                 norm=True)
                 navi_rag_instance.query_to_embedding['passage'][query] = query_emb_np
            
            if isinstance(query_emb_np, np.ndarray):
                query_emb = torch.from_numpy(query_emb_np).float().view(1, -1).to(self.device)
            else:
                query_emb = query_emb_np.view(1, -1).to(self.device)
    
            current_node_emb = navi_rag_instance.get_embedding_for_node(current_node) 
            if current_node_emb is not None:
                current_node_emb = current_node_emb.view(1, -1) 
            else:
                logger.warning(f"Start node {start_node} has NO embedding! Aborting episode.")
                return path
            
            sim_current = F.cosine_similarity(current_node_emb, query_emb, dim=1).item()
    
            for step in range(max_steps):
                try:
                    neighbor_indices = navi_rag_instance.graph.neighbors(navi_rag_instance.node_name_to_vertex_idx[current_node])
                    neighbor_names = [navi_rag_instance.graph.vs[i]["name"] for i in neighbor_indices if navi_rag_instance.graph.vs[i]["name"] not in path]
                    if not neighbor_names: break
                    
                    neighbor_embs_list = [] 
                    valid_neighbor_names = []
                    for n in neighbor_names:
                        emb = navi_rag_instance.get_embedding_for_node(n)
                        if emb is not None:
                            neighbor_embs_list.append(emb)
                            valid_neighbor_names.append(n)
                    
                    if not valid_neighbor_names: break
                    neighbor_names = valid_neighbor_names
                    
                    neighbor_embs = torch.stack(neighbor_embs_list).to(self.device) 
                    
                    history_embs_list = [navi_rag_instance.get_embedding_for_node(n) for n in path]
                    valid_history_embs = [e for e in history_embs_list if e is not None]
                    if not valid_history_embs: continue
                    history_embs = torch.stack(valid_history_embs).unsqueeze(0).to(self.device)
                    
                    state_embs = (current_node_emb, query_emb, history_embs, neighbor_embs)
                    action_info = self.select_action(state_embs)
                    if action_info is None: break
                    action_index, log_prob = action_info
                    
                    next_node = neighbor_names[action_index]
                    next_node_emb = neighbor_embs_list[action_index].view(1, -1) 
                
                except Exception as e:
                    logger.debug(f"run_episode Error: {e}"); break
    
                # --- Reward Calculation ---
                sim_next = F.cosine_similarity(next_node_emb, query_emb, dim=1).item()
                progress_reward = max(0, sim_next - sim_current)
                
                reward = reward_step 
                reward += progress_reward * 1.0
                        
                if next_node in navi_rag_instance.passage_node_keys:
                    passage_content = navi_rag_instance.chunk_embedding_store.get_row(next_node)["content"]
                    if passage_content in gold_docs:
                        reward += reward_gold
                        self.rewards.append(reward)
                        path.append(next_node)
                        break
                        
                self.rewards.append(reward)
    
                # --- State Transition ---
                path.append(next_node)
                current_node = next_node
                current_node_emb = next_node_emb 
                sim_current = sim_next 
    
            return path     

    def update_policy(self):
        """
        Updates the policy network parameters using the REINFORCE algorithm based on collected rewards and log probabilities.
        """
        if not self.saved_log_probs:
            return

        R = 0
        policy_loss_terms = []
        returns = []
        
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = (returns - returns.mean()) 

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss_terms.append(-log_prob * R)

        if not policy_loss_terms:
            return

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss_terms).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        self.rewards = []
        self.saved_log_probs = []

# ==============================================================================
#  NaviRAG Main Class
# ==============================================================================

class NaviRAG:
    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None,
                 *args, **kwargs):
        """
        Initializes the NaviRAG instance with reinforcement learning parameters.
        Sets up the device, working directory, and query triplet cache.
        """
        # --- Base Initialization Logic ---
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"NaviRAG init with config:\n  {_print_config}\n")

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None

        # --- RL Specific Initialization ---
        self.rl_learning_rate = getattr(self.global_config, 'rl_learning_rate', 1e-4)
        self.rl_gamma = getattr(self.global_config, 'rl_gamma', 0.99)
        self.rl_max_episode_steps = getattr(self.global_config, 'rl_max_episode_steps', 15)
        self.rl_hidden_dim = getattr(self.global_config, 'rl_hidden_dim', 256)
        self.rl_reward_gold = getattr(self.global_config, 'rl_reward_gold', 1.0)
        self.rl_reward_step = getattr(self.global_config, 'rl_reward_step', -0.02)
        
        self.beam_width = getattr(self.global_config, 'beam_width', 25)    
        self.w_policy = getattr(self.global_config, 'w_policy', 0.2)
        self.w_sim = getattr(self.global_config, 'w_sim', 0.6)
        self.w_ppr = getattr(self.global_config, 'w_ppr', 0.2)  
        
        self.w_query = getattr(self.global_config, 'w_query', 0.5)
        
        self.rl_policy_network = None
        self.rl_agent = None
        self.policy_network_path = os.path.join(self.working_dir, "best_policy.pth")

        self.device = torch.device("cuda")        
        
        logger.info(f"NaviRAG initialization complete. RL Hyperparams: LR={self.rl_learning_rate}, Gamma={self.rl_gamma}")

        # Initialize query triplet cache
        self.query_cache_path = os.path.join(self.global_config.save_dir, "query_triplets_cache.json")
        self.query_triplets_cache = {}
        
        if os.path.exists(self.query_cache_path):
            try:
                with open(self.query_cache_path, 'r', encoding='utf-8') as f:
                    self.query_triplets_cache = json.load(f)
                logger.info(f"Loaded {len(self.query_triplets_cache)} query triplet cache entries.")
            except Exception as e:
                logger.warning(f"Failed to load query cache: {e}")

    def load_rl_agent(self, training=False):
        """
        Loads the pre-trained policy network or initializes a new one for training.
        Also moves embeddings to the GPU for optimized retrieval performance.
        """
        if self.rl_agent is not None:
            return

        logger.info(f"Loading/Initializing policy network from {self.policy_network_path}...")
        
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
            
        # --- [Optimization] Move all Embeddings to GPU ---
        logger.info("Moving graph embeddings to GPU for training acceleration...")
        if self.entity_embeddings is not None:
            self.entity_embeddings_tensor = torch.from_numpy(self.entity_embeddings).float().to(self.device)
        if self.passage_embeddings is not None:
            self.passage_embeddings_tensor = torch.from_numpy(self.passage_embeddings).float().to(self.device)
        
        self.entity_key_to_idx = {k: i for i, k in enumerate(self.entity_node_keys)}
        self.passage_key_to_idx = {k: i for i, k in enumerate(self.passage_node_keys)}

        if not hasattr(self.embedding_model, 'embedding_dim') or self.embedding_model.embedding_dim is None:
             if self.passage_embeddings is not None and self.passage_embeddings.shape[1] > 0:
                 embedding_dim = self.passage_embeddings.shape[1]
             else:
                raise ValueError("Cannot determine embedding dimension. Ensure embedding_model has 'embedding_dim' or embeddings are loaded.")
        else:
            embedding_dim = self.embedding_model.embedding_dim
            
        logger.info(f"Dynamically retrieved embedding dimension: {embedding_dim}")

        self.rl_policy_network = PolicyNetwork(embedding_dim, self.rl_hidden_dim)
        
        if os.path.exists(self.policy_network_path):
            self.rl_policy_network.load_state_dict(torch.load(self.policy_network_path))
            logger.info("Policy model loaded successfully.")
        elif not training:
            logger.error("No pre-trained policy model found.")
            raise FileNotFoundError(f"Policy file not found at {self.policy_network_path}")
        

        self.rl_policy_network.to(self.device)
        self.rl_agent = RLAgent(self.rl_policy_network, self.rl_learning_rate, self.rl_gamma)
        self.rl_policy_network.train() if training else self.rl_policy_network.eval()        

    def get_embedding_for_node(self, node_key: str) -> Optional[torch.Tensor]:
        """
        Retrieves the corresponding embedding tensor for a given node key (entity or passage) from the GPU cache.
        """
        if hasattr(self, 'entity_embeddings_tensor'):
            idx = self.entity_key_to_idx.get(node_key)
            if idx is not None:
                return self.entity_embeddings_tensor[idx]
        
        if hasattr(self, 'passage_embeddings_tensor'):
            idx = self.passage_key_to_idx.get(node_key)
            if idx is not None:
                return self.passage_embeddings_tensor[idx]
        return None
    
    def run_ppr(self, reset_prob: np.ndarray, damping: float = 0.5) -> np.ndarray:
        """
        Runs Personalized PageRank (PPR) on the knowledge graph using the provided reset probability vector.
        Returns the raw probability array.
        """
        if damping is None: damping = 0.5
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            damping=damping, directed=False, weights='weight', reset=reset_prob, implementation='prpack'
        )
        return np.array(pagerank_scores)

    def extract_query_triplets(self, query: str) -> str:
        """
        Extracts structured triplets (S, R, O) from a query using OpenIE and caches the result.
        """
        # 1. Check cache
        if query in self.query_triplets_cache:
            cache_item = self.query_triplets_cache[query]
            if isinstance(cache_item, dict):
                return cache_item.get("extracted_triplets", "")
            return cache_item

        # 2. Initialize OpenIE
        if not hasattr(self, 'query_openie'):
            try:
                self.query_openie = OpenIE(llm_model=self.llm_model)
            except Exception as e:
                logger.warning(f"OpenIE initialization failed: {e}")
                return ""

        try:
            chunk_key = compute_mdhash_id(content=query, prefix="query_")
            
            # 3. Execute extraction (NER + Triple Extraction)
            ner_output = self.query_openie.ner(chunk_key=chunk_key, passage=query)
            
            triple_output = self.query_openie.triple_extraction(
                chunk_key=chunk_key, 
                passage=query, 
                named_entities=ner_output.unique_entities,
                prompt_name='query_triple_extraction' 
            )
            
            triples = triple_output.triples
            
            # 4. Handle empty results
            if not triples:
                empty_entry = {
                    "original_query": query,
                    "extracted_triplets": "",
                    "raw_triples": []
                }
                self.query_triplets_cache[query] = empty_entry
                return ""
                
            # 5. Format output
            triplet_strings = []
            for t in triples:
                if len(t) < 3: continue
                s, r, o = t[0], t[1], t[2]
                
                parts = []
                if s != "?": parts.append(s)
                if r != "?": parts.append(r)
                if o != "?": parts.append(o)
                
                phrase = " ".join(parts)
                triplet_strings.append(phrase)
            
            result_str = " ".join(triplet_strings)
            
            # 6. Build and save rich cache structure
            cache_entry = {
                "original_query": query,
                "extracted_triplets": result_str,
                "raw_triples": triples 
            }

            self.query_triplets_cache[query] = cache_entry
            
            try:
                with open(self.query_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.query_triplets_cache, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to write query cache: {e}")

            return result_str
            
        except Exception as e:
            logger.warning(f"Exception during query triplet extraction: {e}")
            return ""
            
    def precompute_embeddings(self, queries: List[str]):
        """
        Pre-computes embeddings for raw queries, extracted triplets, and DPR paths in batch to save time during retrieval.
        """
        logger.info(f"Starting pre-computation for {len(queries)} query embeddings...")
        
        # --- 1. Raw Query Encoding (Path A) ---
        raw_queries_to_encode = [q for q in queries if q not in self.query_to_embedding['triple']]
        if raw_queries_to_encode:
            logger.info(f"Encoding {len(raw_queries_to_encode)} raw queries...")
            raw_embeddings = self.embedding_model.batch_encode(
                raw_queries_to_encode,
                instruction=get_query_instruction('query_to_fact'),
                norm=True
            )
            for q, emb in zip(raw_queries_to_encode, raw_embeddings):
                self.query_to_embedding['triple'][q] = emb
        
        # --- 2. [Optimization] Concurrent Triplet Extraction (Path B) ---
        triplets_to_encode = []
        queries_needing_extraction = [q for q in queries if q not in self.query_triplets_cache]
        
        if queries_needing_extraction:
            logger.info(f"Extracting triplets for {len(queries_needing_extraction)} queries...")
            
            def extract_worker(q):
                return q, self.extract_query_triplets(q) 

            with ThreadPoolExecutor() as executor: 
                futures = {executor.submit(extract_worker, q): q for q in queries_needing_extraction}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting query triplets"):
                    q, t_str = future.result()    
                    
        try:
            with open(self.query_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.query_triplets_cache, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        # --- 3. Collect Triplets and Encode ---
        for q in queries:
            triplets_str = self.extract_query_triplets(q)
            if triplets_str and triplets_str not in self.query_to_embedding['triple']:
                triplets_to_encode.append(triplets_str)
        
        unique_triplets = list(set(triplets_to_encode))
        if unique_triplets:
            logger.info(f"Encoding {len(unique_triplets)} unique triplet strings...")
            triplet_embeddings = self.embedding_model.batch_encode(
                unique_triplets,
                instruction=get_query_instruction('fact_to_fact'),
                norm=True
            )
            for t_str, emb in zip(unique_triplets, triplet_embeddings):
                self.query_to_embedding['triple'][t_str] = emb

        # --- 4. DPR Encoding (Path C) ---
        passage_queries = [q for q in queries if q not in self.query_to_embedding['passage']]
        if passage_queries:
            logger.info(f"Encoding {len(passage_queries)} queries for DPR...")
            p_embeddings = self.embedding_model.batch_encode(
                passage_queries,
                instruction=get_query_instruction('query_to_passage'),
                norm=True
            )
            for q, emb in zip(passage_queries, p_embeddings):
                self.query_to_embedding['passage'][q] = emb

        logger.info("All embedding pre-computations finished!")

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Calculates fusion scores using dual-path recall: raw query similarity and triplet-based similarity.
        """
        # --- Path A: Raw Query ---
        query_emb_A = self.query_to_embedding['triple'].get(query, None)
        if query_emb_A is None:
            query_emb_A = self.embedding_model.batch_encode(query, 
                                                            instruction=get_query_instruction('query_to_fact'), 
                                                            norm=True)
            self.query_to_embedding['triple'][query] = query_emb_A

        if len(self.fact_embeddings) == 0:
            return np.array([])

        scores_A = np.dot(self.fact_embeddings, query_emb_A.T)
        scores_A = np.squeeze(scores_A) if scores_A.ndim == 2 else scores_A
        scores_A = min_max_normalize(scores_A)

        # --- Path B: Structured Triplets ---
        triplets_str = self.extract_query_triplets(query)
        
        if not triplets_str:
            return scores_A
            
        query_emb_B = self.query_to_embedding['triple'].get(triplets_str, None)
        if query_emb_B is None:
            query_emb_B = self.embedding_model.batch_encode(triplets_str, 
                                                            instruction=get_query_instruction('fact_to_fact'), 
                                                            norm=True)
            self.query_to_embedding['triple'][triplets_str] = query_emb_B
            
        scores_B = np.dot(self.fact_embeddings, query_emb_B.T)
        scores_B = np.squeeze(scores_B) if scores_B.ndim == 2 else scores_B
        scores_B = min_max_normalize(scores_B)
        
        w_query = self.w_query 
        final_scores = w_query * scores_A + (1 - w_query) * scores_B
        
        return final_scores
            
    def _get_ppr_inputs_and_scores(self, query, top_k_facts, top_k_fact_indices):
        """
        Helper method to calculate node weights for PPR and the resulting PPR scores for all nodes.
        """
        passage_node_weight = getattr(self.global_config, 'passage_node_weight', 0.05)
        
        linking_score_map, phrase_scores = {}, {}
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        query_fact_scores = self.get_fact_scores(query)
        for rank, f in enumerate(top_k_facts):
            subject_phrase, object_phrase = f[0].lower(), f[2].lower()
            fact_score = query_fact_scores[top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                if phrase_key in self.node_name_to_vertex_idx:
                    phrase_id = self.node_name_to_vertex_idx[phrase_key]
                    current_score = phrase_weights[phrase_id]
                    new_score = fact_score
                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        new_score /= len(self.ent_node_to_chunk_ids[phrase_key])
                    phrase_weights[phrase_id] = max(current_score, new_score) 
                if phrase not in phrase_scores: phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)
        
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))
        
        if getattr(self.global_config, 'linking_top_k', None):
            phrase_weights, _ = self.get_top_k_weights(self.global_config.linking_top_k, phrase_weights, linking_score_map)

        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)
        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            if passage_node_key in self.node_name_to_vertex_idx:
                passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
                passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight

        node_weights = phrase_weights + passage_weights
        
        all_node_pagerank_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        all_node_names = self.graph.vs["name"]
        ppr_scores_dict = {name: score for name, score in zip(all_node_names, all_node_pagerank_scores)}
        
        return node_weights, ppr_scores_dict

def retrieve(self, queries: List[str], num_to_retrieve: int = None, gold_docs: List[List[str]] = None) -> Union[List[QuerySolution], Tuple[List[QuerySolution], Dict]]:
        """
        Retrieves relevant passages for a set of queries using strictly the 'rl_rerank' mode.
        """
        # Always load the RL agent since we are strictly in rl_rerank mode
        if self.rl_agent is None:
            self.load_rl_agent(training=False)

        if num_to_retrieve is None: num_to_retrieve = self.global_config.retrieval_top_k
        if gold_docs is not None: retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)
        if not self.ready_to_retrieve: self.prepare_retrieval_objects()

        self.precompute_embeddings(queries)
        
        retrieve_start_time = time.time()
        retrieval_results = []
        desc = "Retrieving with NaviRAG"

        for q_idx, query in tqdm(enumerate(queries), desc=desc, total=len(queries)):
                
            query_fact_scores = self.get_fact_scores(query)
            
            top_k_fact_indices, top_k_facts = self.rank_facts(query, query_fact_scores)

            if not top_k_facts:
                # Fallback to DPR if no facts are found
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                # Execute RL Beam Search
                passage_info, query_emb_raw = self.run_rl_graph_search_rerank(query, top_k_facts)
                
                # Ensure query embedding is a tensor on the correct device
                if isinstance(query_emb_raw, np.ndarray):
                    query_emb_tensor = torch.from_numpy(query_emb_raw).float().to(self.device)
                else:
                    query_emb_tensor = query_emb_raw.to(self.device)                

                if not passage_info:
                    logger.warning("RL Search found no passages, falling back to DPR.")
                    sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                else:
                    # 'rl_rerank' Logic: Combine Policy Score + Cosine Similarity + PPR Score
                    
                    # 1. Get PPR Scores for the context
                    _, ppr_scores_dict = self._get_ppr_inputs_and_scores(query, top_k_facts, top_k_fact_indices)
                    
                    final_scores = {}
                    for passage, info in passage_info.items():
                        path_score = info['score']
                        passage_emb_tensor = self.get_embedding_for_node(passage)
                        if passage_emb_tensor is None: continue
                        
                        # 2. Calculate Cosine Similarity
                        similarity_score = F.cosine_similarity(
                            query_emb_tensor.view(1, -1), 
                            passage_emb_tensor.view(1, -1)
                        ).item()
                        
                        # 3. Get PPR Score
                        ppr_score = ppr_scores_dict.get(passage, 0.0)

                        # 4. Weighted Sum
                        final_scores[passage] = self.w_policy * path_score + self.w_sim * similarity_score + self.w_ppr * ppr_score

                    # Sort passages by final score
                    final_sorted_passages = sorted(final_scores, key=final_scores.get, reverse=True)
                    passage_key_to_idx = {key: i for i, key in enumerate(self.passage_node_keys)}
                    
                    # Map back to internal IDs
                    sorted_doc_ids = np.array([passage_key_to_idx[p] for p in final_sorted_passages if p in passage_key_to_idx])
                    sorted_doc_scores = np.array([final_scores[p] for p in final_sorted_passages if p in passage_key_to_idx])

            # Retrieve actual content based on sorted IDs
            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]
            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        all_retrieval_time = time.time() - retrieve_start_time
        
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, _ = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[r.docs for r in retrieval_results], k_list=k_list)
            
            logger.info(f"Total Retrieval Time: {all_retrieval_time:.2f}s")
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")
            return retrieval_results, overall_retrieval_result
            
        return retrieval_results


    def rag_qa(self, queries: List[Union[str, QuerySolution]], gold_docs: List[List[str]] = None, gold_answers: List[List[str]] = None):
        """
        Executes the full RAG pipeline: Retrieval followed by Question Answering, including evaluation against gold answers.
        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)
        
        overall_retrieval_result = {}
        if not isinstance(queries[0], QuerySolution):
            retrieval_output = self.retrieve(queries=queries, gold_docs=gold_docs)
            if gold_docs is not None:
                queries, overall_retrieval_result = retrieval_output
            else:
                queries = retrieval_output
        
        queries_solutions, all_response_message, all_metadata = self.qa(queries)
        
        if gold_answers is not None:
            overall_qa_em_result, _ = qa_em_evaluator.calculate_metric_scores(gold_answers=gold_answers, predicted_answers=[q.answer for q in queries_solutions], aggregation_fn=np.max)
            overall_qa_f1_result, _ = qa_f1_evaluator.calculate_metric_scores(gold_answers=gold_answers, predicted_answers=[q.answer for q in queries_solutions], aggregation_fn=np.max)
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_em_result.items()}
       
            logger.info(f"Evaluation results for QA: {overall_qa_results}")
            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        
        return queries_solutions, all_response_message, all_metadata

    def run_rl_graph_search_rerank(self, query: str, top_k_facts: List[Tuple]) -> Tuple[Dict, torch.Tensor]:
        """
        Executes RL-based beam search on the graph to find high-probability paths leading to relevant passages.
        """
        beam_width = self.beam_width
        w_policy = self.w_policy
        w_sim = self.w_sim

        # 1. Get Seeds
        seed_nodes = set()
        for fact in top_k_facts:
            for entity in [fact[0], fact[2]]:
                entity_key = compute_mdhash_id(content=entity.lower(), prefix="entity-")
                if entity_key in self.entity_node_keys:
                    seed_nodes.add(entity_key)
        
        # 2. Prepare Query Embedding
        query_emb_obj = self.query_to_embedding['passage'].get(query, None)
        if query_emb_obj is None:
            query_emb_obj = self.embedding_model.batch_encode(query, 
                                                             instruction=get_query_instruction('query_to_passage'), 
                                                             norm=True)
            self.query_to_embedding['passage'][query] = query_emb_obj

        if isinstance(query_emb_obj, np.ndarray):
            query_emb = torch.from_numpy(query_emb_obj).float().to(self.device)
        else:
            query_emb = query_emb_obj.to(self.device)
        query_emb = query_emb.view(1, -1)

        # 3. Validate seeds and initialize beam
        valid_seeds = [s for s in seed_nodes if self.get_embedding_for_node(s) is not None]
        
        if not valid_seeds:
            logger.warning("No valid RL starting seeds found, falling back to DPR.")
            dpr_fallback = {}
            dpr_ids, dpr_scores = self.dense_passage_retrieval(query)
            for doc_id, score in zip(dpr_ids, dpr_scores):
                node_key = self.passage_node_keys[doc_id]
                dpr_fallback[node_key] = {'score': float(score), 'path': []}
            return dpr_fallback, query_emb

        active_beams = [{'path': [seed], 'score': 0.0} for seed in valid_seeds]
        passage_info = defaultdict(lambda: {'score': -float('inf'), 'path': []})

        # --- 4. Beam Search Loop ---
        for step in range(self.rl_max_episode_steps):
            if not active_beams: break
            
            all_candidate_beams = []
            expanded_nodes_in_step = set()

            for beam in active_beams:
                current_node = beam['path'][-1]
                if current_node in expanded_nodes_in_step: continue
                expanded_nodes_in_step.add(current_node)

                try:
                    with torch.no_grad():
                        # A. Get neighbors
                        neighbor_indices = self.graph.neighbors(self.node_name_to_vertex_idx[current_node])
                        neighbor_names = [self.graph.vs[i]["name"] for i in neighbor_indices if self.graph.vs[i]["name"] not in beam['path']]
                        if not neighbor_names: continue
                        
                        # B. Get neighbor embeddings
                        neighbor_embs_list = []
                        valid_neighbor_names = []
                        for n in neighbor_names:
                            emb = self.get_embedding_for_node(n)
                            if emb is not None:
                                neighbor_embs_list.append(emb)
                                valid_neighbor_names.append(n)
                        if not valid_neighbor_names: continue
                        
                        neighbor_embs = torch.stack(neighbor_embs_list) 

                        # C. Get current and history embeddings
                        current_node_emb = self.get_embedding_for_node(current_node).view(1, -1)
                        
                        history_embs_list = [self.get_embedding_for_node(n) for n in beam['path']]
                        valid_history_embs = [e for e in history_embs_list if e is not None]
                        if not valid_history_embs: continue
                        history_embs = torch.stack(valid_history_embs).unsqueeze(0)

                        # D. Policy Network Forward Pass
                        dist = self.rl_policy_network(current_node_emb, query_emb, history_embs, neighbor_embs)
                        if dist.probs.nelement() == 0: continue

                        # E. Expand Beams
                        for i, next_node in enumerate(valid_neighbor_names):
                            policy_score = dist.log_prob(torch.tensor(i).to(self.device)).item()
                            
                            new_path_score = beam['score'] + w_policy * policy_score
                            
                            new_path = beam['path'] + [next_node]
                            all_candidate_beams.append({'path': new_path, 'score': new_path_score})
                            
                            if next_node in self.passage_node_keys:
                                if new_path_score > passage_info[next_node]['score']:
                                    passage_info[next_node]['score'] = new_path_score
                                    passage_info[next_node]['path'] = new_path

                except Exception as e:
                    logger.debug(f"Beam Search Error (Node: {current_node}): {e}")
                    continue 
            
            if not all_candidate_beams: break
            
            # Pruning
            active_beams = sorted(all_candidate_beams, key=lambda x: x['score'], reverse=True)[:beam_width]

        # --- 5. Result check ---
        if not passage_info:
            logger.warning("RL Beam Search reached no passage nodes, falling back to DPR.")
            dpr_fallback = {}
            dpr_ids, dpr_scores = self.dense_passage_retrieval(query)
            for doc_id, score in zip(dpr_ids, dpr_scores):
                node_key = self.passage_node_keys[doc_id]
                dpr_fallback[node_key] = {'score': float(score), 'path': []}
            return dpr_fallback, query_emb

    def rank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """
        Ranks candidate facts based on initial similarity scores to select top-k facts for linking.
        """
        link_top_k: int = self.global_config.linking_top_k
        
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rank': [], 'facts_after_rank': []}
            
        try:
            if len(query_fact_scores) <= link_top_k:
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            top_k_fact_indices = candidate_fact_indices
            top_k_facts = candidate_facts
                     
            return top_k_fact_indices, top_k_facts
            
        except Exception as e:
            logger.error(f"Error in rank_facts: {str(e)}")
            return [], [], {'facts_before_rank': [], 'facts_after_rank': [], 'error': str(e)}

    def analyze_multihop_capability(self, test_set_path: str, dataset_name: str = "musique"):
        """
        Specialized analysis function that loads a test set, groups items by hop count, and evaluates Recall@10.
        """
        logger.info(f"Starting Multi-hop Capability analysis (Dataset: {dataset_name})...")
        
        with open(test_set_path, 'r') as f:
            data = json.load(f)
            
        hop_groups = {'2-hop': [], '3-hop': [], '4-hop': []}
        
        for item in data:
            num_hops = 0
            if 'contexts' in item: # Musique format
                num_hops = sum(1 for p in item['contexts'] if p.get('is_supporting', False))
            elif 'supporting_facts' in item: # 2Wiki/Hotpot format
                num_hops = len(item['supporting_facts'])
            
            if num_hops == 2:
                hop_groups['2-hop'].append(item)
            elif num_hops == 3:
                hop_groups['3-hop'].append(item)
            elif num_hops >= 4:
                hop_groups['4-hop'].append(item)
        
        logger.info(f"Grouping Statistics: 2-hop: {len(hop_groups['2-hop'])}, 3-hop: {len(hop_groups['3-hop'])}, 4-hop: {len(hop_groups['4-hop'])}")
            
        if self.rl_agent is None:
            self.load_rl_agent(training=False)
        
        results = {}
        all_queries = [item['question'] for item in data]
        self.precompute_embeddings(all_queries)
        
        evaluator = RetrievalRecall(global_config=self.global_config)
        
        for hop, items in hop_groups.items():
            if not items: continue
            logger.info(f"Evaluating {hop} ({len(items)} samples)...")
            
            queries = [i['question'] for i in items]
            
            gold_docs = []
            for sample in items:
                if 'supporting_facts' in sample:
                    gold_title = set([item[0] for item in sample['supporting_facts']])
                    gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
                    if dataset_name and dataset_name.startswith('hotpotqa'):
                        gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
                    else:
                        gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
                elif 'contexts' in sample:
                    gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
                else:
                    assert 'paragraphs' in sample, "`paragraphs` should be in sample"
                    gold_paragraphs = [p for p in sample['paragraphs'] if p.get('is_supporting', True)]
                    gold_doc = [item['title'] + '\n' + (item.get('text', item.get('paragraph_text'))) for item in gold_paragraphs]
                gold_docs.append(list(set(gold_doc)))

            retrieved_results = self.retrieve(queries=queries, num_to_retrieve=10)
            
            score_dict, _ = evaluator.calculate_metric_scores(
                gold_docs=gold_docs, 
                retrieved_docs=[r.docs for r in retrieved_results], 
                k_list = [1, 2, 5, 10]
            )
            results[hop] = score_dict
            
            logger.info(f">>> {hop} Result: {results[hop]}")

        logger.info("\n=== Final Hop Analysis Results ===")
        logger.info(json.dumps(results, indent=2))
        return results    

    # ==============================================================================
    #  Base Functionality (Indexing, Graph Ops, QA)
    # ==============================================================================

    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the NaviRAG framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict)

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the NaviRAG class.
        """

        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        #Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        #Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        #Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        #Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.
        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.
        """
        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.
        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.
        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            # If the graph is empty but we have nodes, we need to add them
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # Check if all entity and passage nodes are in the graph
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                # If nodes are missing, rebuild the graph
                self.add_new_nodes()
                self.save_igraph()
                # Update the mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # Check if the lengths match
            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")
                
                # If there are missing keys, create empty entries for them
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )

            # prepare data_store
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    