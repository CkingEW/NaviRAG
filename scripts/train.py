import os
import json
import argparse
import logging
import random
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import from navirag module
from navirag.NaviRAG import NaviRAG
from navirag.utils.config_utils import BaseConfig
from navirag.utils.misc_utils import string_to_bool, compute_mdhash_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_gold_docs(samples: List[Dict], dataset_name: str = None) -> List[List[str]]:
    gold_docs = []
    for sample in samples:
        doc_set = set()
        if 'supporting_facts' in sample:
            gold_titles = set(item[0] for item in sample['supporting_facts'])
            for item in sample['context']:
                if item[0] in gold_titles:
                    text = ''.join(item[1]) if dataset_name and dataset_name.startswith('hotpotqa') else ' '.join(item[1])
                    doc_set.add(item[0] + '\n' + text)
        elif 'contexts' in sample:
            for item in sample['contexts']:
                if item['is_supporting']:
                    doc_set.add(item['title'] + '\n' + item['text'])
        elif 'paragraphs' in sample:
            for p in sample['paragraphs']:
                if p.get('is_supporting', True):
                    doc_set.add(p['title'] + '\n' + p.get('text', p.get('paragraph_text', '')))
        else:
            logger.warning(f"Sample missing valid answer source fields: {sample.get('id', 'N/A')}")
        gold_docs.append(list(doc_set))
    return gold_docs

def precompute_gold_embeddings(navirag: NaviRAG, all_data_points: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Precompute embeddings for all gold documents in the training/validation sets.
    """
    logger.info("Precomputing embeddings for all gold documents...")
    
    # 1. Collect all unique document texts
    unique_gold_texts = set()
    for item in all_data_points:
        # item['gold_docs'] is a list of strings
        for doc_text in item['gold_docs']:
            unique_gold_texts.add(doc_text)
            
    sorted_texts = list(unique_gold_texts)
    logger.info(f"Total of {len(sorted_texts)} unique gold documents to encode.")
    
    if not sorted_texts:
        return {}

    # 2. Batch encode
    # No instruction needed for documents, or use empty string
    embeddings_np = navirag.embedding_model.batch_encode(
        sorted_texts, 
        instruction="", 
        norm=True
    )
    
    # 3. Build lookup dictionary
    cache = {}
    for text, emb in zip(sorted_texts, embeddings_np):

        cache[text] = torch.from_numpy(emb).float()
        
    logger.info("Gold document embedding precomputation completed.")
    return cache


def prepare_datasets(dataset_name: str, base_dir: str, val_split: float, test_split: float, random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Check if dataset splits exist; if not, create and save them.
    Ensures that for small datasets, each split has at least one sample.
    Returns training and validation samples.
    """
    train_path = os.path.join(base_dir, "train.json")
    val_path = os.path.join(base_dir, "val.json")
    test_path = os.path.join(base_dir, "test.json")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        logger.info("Found existing dataset splits, loading directly...")
        with open(train_path, "r") as f:
            train_samples = json.load(f)
        with open(val_path, "r") as f:
            val_samples = json.load(f)
        
        # Add check: if loaded files are empty, warn user
        if not train_samples or not val_samples:
             logger.warning("Loaded dataset splits contain empty training or validation sets. This may cause issues.")
        else:
            logger.info(f"Loaded: {len(train_samples)} training samples, {len(val_samples)} validation samples.")
            logger.info(f"Test set saved at {test_path}, will not be used during training.")
        return train_samples, val_samples

    logger.info("No dataset splits found, creating new splits...")
    original_data_path = f"datasets/{dataset_name}/{dataset_name}.json"
    if not os.path.exists(original_data_path):
        raise FileNotFoundError(f"Original dataset file not found: {original_data_path}")

    with open(original_data_path, "r") as f:
        all_samples = json.load(f)

    random.seed(random_seed)
    random.shuffle(all_samples)

    total_size = len(all_samples)
    
    # Handle edge cases for very small datasets
    if total_size < 3:
        logger.warning(f"Dataset size ({total_size}) is too small to split into train/val/test.")
        train_samples = all_samples
        val_samples = all_samples
        test_samples = all_samples
        logger.warning("All samples will be used for training.")

    else:
        # **Ensure test and validation sets have at least one sample**
        test_size = max(1, int(total_size * test_split))
        val_size = max(1, int(total_size * val_split))
        
        # Ensure training set also has at least one sample
        if total_size - test_size - val_size < 1:
            # If not enough for training, prioritize test and val, then borrow from val for train
            val_size = max(1, val_size - 1)
            if total_size - test_size - val_size < 1:
                # Extreme case: still not enough, borrow from test
                test_size = max(1, test_size - 1)

        train_size = total_size - test_size - val_size
        if train_size < 1:
             raise ValueError(f"Dataset size ({total_size}) is too small to maintain at least one sample for Test ({test_size}) and Val ({val_size}) while keeping samples for Train.")

        test_samples = all_samples[:test_size]
        val_samples = all_samples[test_size : test_size + val_size]
        train_samples = all_samples[test_size + val_size :]

    # --- Ensure directory exists and save files ---
    os.makedirs(base_dir, exist_ok=True)

    with open(test_path, "w") as f:
        json.dump(test_samples, f, indent=4)
    logger.info(f"Test set saved to {test_path} ({len(test_samples)} samples)")

    with open(val_path, "w") as f:
        json.dump(val_samples, f, indent=4)
    logger.info(f"Validation set saved to {val_path} ({len(val_samples)} samples)")

    with open(train_path, "w") as f:
        json.dump(train_samples, f, indent=4)
    logger.info(f"Training set saved to {train_path} ({len(train_samples)} samples)")

    return train_samples, val_samples


def train_rl_agent_epoch(navirag: NaviRAG, training_data: List[Dict], epoch_num: int, gold_emb_cache: Dict[str, torch.Tensor]):
    agent = navirag.rl_agent
    total_epoch_reward = 0
    progress_bar = tqdm(training_data, desc=f"Epoch {epoch_num} [Training]")
    
    # Ensure model is in training mode
    navirag.rl_policy_network.train()

    for data_point in progress_bar:
        query, gold_docs = data_point['query'], data_point['gold_docs']

        # --- Get Gold Doc Embeddings from Cache ---
        gold_doc_embs = None
        if gold_docs:
            embs_list = []
            for doc in gold_docs:
                if doc in gold_emb_cache:
                    embs_list.append(gold_emb_cache[doc])
            
            if embs_list:
                gold_doc_embs = torch.stack(embs_list) # Stack into (N, dim)
        
        # --- Get Seed Nodes ---
        query_fact_scores = navirag.get_fact_scores(query)
        _, top_k_facts, _ = navirag.rerank_facts(query, query_fact_scores)
        seed_nodes = set()
        for fact in top_k_facts:
            for entity in [fact[0], fact[2]]:
                entity_key = compute_mdhash_id(content=entity.lower(), prefix="entity-")
                if entity_key in navirag.entity_node_keys: 
                    seed_nodes.add(entity_key)
        
        if not seed_nodes: continue
        
        valid_seeds = [s for s in seed_nodes if navirag.get_embedding_for_node(s) is not None]
        if not valid_seeds: continue
        start_node = random.choice(valid_seeds)
        
        # --- Run Episode and Update ---
        agent.run_episode(
            navi_rag_instance=navirag,
            start_node=start_node,
            query=query,
            gold_docs=gold_docs,
            gold_doc_embs=gold_doc_embs,
            max_steps=navirag.rl_max_episode_steps,
            reward_gold=navirag.rl_reward_gold,
            reward_step=navirag.rl_reward_step
        )
        
        total_epoch_reward += sum(agent.rewards)
        agent.update_policy()
        progress_bar.set_postfix({"Avg Reward": f"{total_epoch_reward / (progress_bar.n + 1):.2f}"})

    avg_reward = total_epoch_reward / len(training_data) if training_data else 0
    logger.info(f"Epoch {epoch_num} [Training] Completed. Avg Reward: {avg_reward:.2f}")
    return avg_reward


def main():
    parser = argparse.ArgumentParser(description="Train NaviRAG RL Agent")
    parser.add_argument('--dataset', type=str, default='2wikimultihopqa', help='Dataset name for training')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='meta/llama-3.3-70b-instruct', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='BAAI/bge-m3 ', help='Embedding model name')
    parser.add_argument('--force_index_from_scratch', type=str, default='false', help='Force re-indexing')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for RL training')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory for index, models, and data splits')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data to use for validation (e.g., 0.1 for 10%)')
    parser.add_argument('--test_split', type=float, default=0.5, help='Fraction of data to use for the final test set')
    parser.add_argument('--validate_every_n_epochs', type=int, default=5, help='Run validation every N epochs')
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = os.path.join(args.save_dir, dataset_name)

    # --- 1. Prepare Datasets ---
    dataset_output_dir = f"datasets/{dataset_name}"
    train_samples, val_samples = prepare_datasets(
        dataset_name=dataset_name,
        base_dir=dataset_output_dir,
        val_split=args.val_split,
        test_split=args.test_split
    )

    # Load Corpus
    corpus_path = f"datasets/{dataset_name}/{dataset_name}_corpus.json"
    
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus file not found: {corpus_path}")
        return

    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    logger.info(f"Corpus loaded, total {len(docs)} documents.")

    # Prepare Data Structures for Training/Validation
    train_queries = [s['question'] for s in train_samples]
    train_gold_docs = get_gold_docs(train_samples, dataset_name)
    training_data = [{'query': q, 'gold_docs': gd} for q, gd in zip(train_queries, train_gold_docs)]
    
    val_queries = [s['question'] for s in val_samples]
    val_gold_docs = get_gold_docs(val_samples, dataset_name)

    # --- 2. Initialize NaviRAG and Index ---
    
    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=string_to_bool(args.force_index_from_scratch),
        force_openie_from_scratch=False, # Re-use OpenIE results during training usually
        retrieval_top_k=200,
        linking_top_k=10,
        qa_top_k=10,
        embedding_batch_size=8,
        openie_mode='online' 
    )
    
    # Updated Class usage: NaviRAG
    navirag = NaviRAG(global_config=config)
    navirag.index(docs)
    navirag.prepare_retrieval_objects()
    navirag.load_rl_agent(training=True)

    all_queries_to_embed = list(set(train_queries + val_queries))
    logger.info(f"Precomputing embeddings for {len(all_queries_to_embed)} unique queries and triplets...")
    navirag.precompute_embeddings(all_queries_to_embed)

    # Embed Gold Docs
    all_data_for_precompute = training_data + [{'query': q, 'gold_docs': gd} for q, gd in zip(val_queries, val_gold_docs)]
    gold_emb_cache = precompute_gold_embeddings(navirag, all_data_for_precompute)

    # --- 3. Training and Validation Loop ---
    best_val_recall10 = -1.0
    best_epoch = -1
    best_model_path = os.path.join(navirag.working_dir, "best_policy.pth")

    logger.info("--- Starting Training and Validation Loop ---")
    for epoch in range(1, args.epochs + 1):
        
        train_rl_agent_epoch(navirag, training_data, epoch, gold_emb_cache)
        
        if epoch % args.validate_every_n_epochs == 0:
            logger.info(f"--- Epoch {epoch}: Evaluating on Validation Set ---")
            
            navirag.rl_policy_network.eval() 
            
            val_metrics = {}
            with torch.no_grad():
                # Removed 'retrieval_mode' argument as per updated NaviRAG.py
                _, retrieval_metrics = navirag.retrieve(
                    queries=val_queries,
                    gold_docs=val_gold_docs
                )
                val_metrics = retrieval_metrics
            
            current_val_recall10 = val_metrics.get('Recall@10', 0.0)
            logger.info(f"Validation Recall@10: {current_val_recall10:.4f}")

            if current_val_recall10 >= best_val_recall10:
                best_val_recall10 = current_val_recall10
                best_epoch = epoch
                torch.save(navirag.rl_policy_network.state_dict(), best_model_path)
                logger.info(f"*** New Best Model Found! Saved to {best_model_path} (Epoch {best_epoch}, Recall@10: {best_val_recall10:.4f}) ***")
            
            navirag.rl_policy_network.train()

        # Save "latest" model every 10 epochs
        if epoch % 10 == 0:
             latest_model_path = os.path.join(navirag.working_dir, "latest_policy.pth")
             torch.save(navirag.rl_policy_network.state_dict(), latest_model_path)
             logger.info(f"Saved latest model checkpoint to {latest_model_path}")


    logger.info(f"Training Completed. Best model was at Epoch {best_epoch} with Validation Recall@10: {best_val_recall10:.4f}")
    logger.info(f"Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main()