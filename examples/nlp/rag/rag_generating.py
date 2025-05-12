# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from llama_index.core import Settings, StorageContext, load_index_from_storage

from nemo.collections.nlp.models.rag.custom_bert_embedder import NeMoBertEmbeddings
from nemo.collections.nlp.models.rag.custom_gpt_llm import NeMoGPTLLM
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="rag_generating")
def main(cfg) -> None:

    # load LLM
    logging.info("Loading LLM.")
    model_path = cfg.generating.llm.model_path
    if cfg.generating.llm.model_type == "gpt":
        Settings.llm = NeMoGPTLLM(model_path=model_path, cfg=cfg)
    else:
        assert cfg.generating.model_type in ["gpt"], "Currently RAG pipeline supports 'gpt' for LLM models."

    # load embedder
    logging.info("Loading embedder.")
    model_path = cfg.indexing.embedder.model_path
    if cfg.indexing.embedder.model_type == "bert":
        embed_model = NeMoBertEmbeddings(model_path=model_path, cfg=cfg)
    else:
        assert cfg.indexing.model_type in ["bert"], "Currently RAG pipeline supports 'bert' for embeddings models."
        embed_model = None
    Settings.embed_model = embed_model

    # load index from disk
    logging.info("Loading index from disk.")
    index_path = cfg.indexing.index_path
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)

    # set query
    logging.info("Setting query.")
    query = cfg.generating.query
    logging.info("Query: ", query)

    # query and print response
    logging.info("Responding to query using relevant contexts.")
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(query)
    logging.info(response)


if __name__ == '__main__':
    main()
