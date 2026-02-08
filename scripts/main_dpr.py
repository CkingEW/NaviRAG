import os
from typing import List, Set, Any
import json
import argparse
import logging
import sys

# Ensure imports work from the root directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from navirag.StandardRAG import StandardRAG
from navirag.utils.misc_utils import string_to_bool
from navirag.utils.config_utils import BaseConfig
from navirag.embedding_store import EmbeddingStore

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gold_docs(samples: List[Any], dataset_name: str = None) -> List[List[str]]:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name and dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            # Fallback for datasets with paragraphs (e.g. Musique)
            gold_paragraphs = []
            if 'paragraphs' in sample:
                for item in sample['paragraphs']:
                    if 'is_supporting' in item and item['is_supporting'] is False:
                        continue
                    gold_paragraphs.append(item)
                gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]
            else:
                gold_doc = []
                
        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples: List[Any]) -> List[Set[str]]:
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'answers' in sample:
            gold_ans = sample['answers']            
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
            
        if gold_ans is not None:
            if isinstance(gold_ans, str):
                gold_ans = [gold_ans]
            assert isinstance(gold_ans, list)
            gold_ans = set(gold_ans)
            if 'answer_aliases' in sample:
                gold_ans.update(sample['answer_aliases'])
            gold_answers.append(gold_ans)
        else:
            gold_answers.append(set())

    return gold_answers

def main():
    parser = argparse.ArgumentParser(description="StandardRAG (DPR) Retrieval and QA")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='Embedding model name')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore all existing storage files and rebuild index from scratch.')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false', 
                        help='If set to False, will try to reuse existing OpenIE results for the corpus.')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
                        help="OpenIE mode: 'offline' uses VLLM batch mode, 'online' uses standard API calls.")
    parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory')
    args = parser.parse_args()

    dataset_name = args.dataset
    # Construct save directory path
    if args.save_dir == 'outputs':
        save_dir = os.path.join(args.save_dir, dataset_name)
    else:
        save_dir = f"{args.save_dir}_{dataset_name}"

    # Load Corpus
    corpus_path = f"datasets/{dataset_name}/{dataset_name}_corpus.json"
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus not found at {corpus_path}")
        return

    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    # Load Test Data
    samples_path = f"datasets/{dataset_name}/test.json"
    if not os.path.exists(samples_path):
        logger.error(f"Test samples not found at {samples_path}")
        return
        
    with open(samples_path, "r") as f:
        samples = json.load(f)
    
    all_queries = [s['question'] for s in samples]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        if len(gold_docs) != len(all_queries):
             logger.warning("Mismatch in length between gold docs and queries.")
    except Exception as e:
        logger.warning(f"Could not load gold docs: {e}")
        gold_docs = None

    # Configure StandardRAG
    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        # Updated path to match navirag structure
        rerank_dspy_file_path="navirag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=10,
        max_qa_steps=3,
        qa_top_k=10,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode
    )

    logger.info("Initializing StandardRAG (DPR Baseline)...")
    
    standard_rag = StandardRAG(global_config=config)

    # Index Documents (StandardRAG mainly embeds chunks)
    standard_rag.index(docs)

    # Execute Retrieval and QA
    standard_rag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)

if __name__ == "__main__":
    main()