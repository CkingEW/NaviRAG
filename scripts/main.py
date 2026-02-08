import os
import json
import argparse
import logging
from typing import List, Set, Union, Any
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from navirag.NaviRAG import NaviRAG
from navirag.utils.config_utils import BaseConfig

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gold_docs(samples: List[Any], dataset_name: str = None) -> List[List[str]]:
    gold_docs = []
    for sample in samples:
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
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

def main():
    parser = argparse.ArgumentParser(description="Running NaviRAG")
    parser.add_argument('--dataset', type=str, default='2wikimultihopqa', help='Dataset name for evaluation')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='meta/llama-4-maverick-17b-128e-instruct', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='Qwen/Qwen3-Embedding-4B ', help='Embedding model name')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Directory where index and RL model are saved')

    
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = os.path.join(args.save_dir, dataset_name)

    corpus_path = f"datasets/{dataset_name}/{dataset_name}_corpus.json"
    
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus file not found at {corpus_path}")
        return

    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    
    logger.info(f"--- Preparing evaluation data for {dataset_name} ---")

    # 1. Load test samples
    samples_path = f"datasets/{dataset_name}/test.json"
    
    if not os.path.exists(samples_path):
        logger.error(f"Test samples file not found at {samples_path}")
        return

    with open(samples_path, "r") as f:
        samples = json.load(f)
    
    all_queries = [s['question'] for s in samples]
    gold_answers = get_gold_answers(samples)
    gold_docs = get_gold_docs(samples, dataset_name)

    logger.info(f"Evaluation data loaded. Total queries: {len(all_queries)}.")

    
    # 2. Configure and Initialize NaviRAG
    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=False, 
        force_openie_from_scratch=False,
        retrieval_top_k=200,
        max_qa_steps=3,
        linking_top_k=10,
        qa_top_k=10,
        max_new_tokens=None,
        graph_type="facts_and_sim_passage_node_unidirectional",
        corpus_len=len(corpus),
        embedding_batch_size=8,
        openie_mode='online'
    )


    navirag = NaviRAG(global_config=config)

    navirag.index(docs)

    # 3. Execute Retrieval and QA
    logger.info(f"--- Starting NaviRAG Retrieval and QA Evaluation ---")

    navirag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)
      

if __name__ == "__main__":
    main()