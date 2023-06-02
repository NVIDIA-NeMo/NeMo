# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


from nemo.collections.nlp.models.duplex_text_normalization import (
    DuplexDecoderModel,
    DuplexTaggerModel,
    DuplexTextNormalizationModel,
)
from nemo.collections.nlp.models.entity_linking.entity_linking_model import EntityLinkingModel
from nemo.collections.nlp.models.glue_benchmark.glue_benchmark_model import GLUEModel
from nemo.collections.nlp.models.information_retrieval import BertDPRModel, BertJointIRModel
from nemo.collections.nlp.models.intent_slot_classification import (
    IntentSlotClassificationModel,
    MultiLabelIntentSlotClassificationModel,
)
from nemo.collections.nlp.models.language_modeling import MegatronGPTPromptLearningModel
from nemo.collections.nlp.models.language_modeling.bert_lm_model import BERTLMModel
from nemo.collections.nlp.models.language_modeling.transformer_lm_model import TransformerLMModel
from nemo.collections.nlp.models.machine_translation import MTEncDecModel
from nemo.collections.nlp.models.question_answering.qa_model import QAModel
from nemo.collections.nlp.models.spellchecking_asr_customization import SpellcheckingAsrCustomizationModel
from nemo.collections.nlp.models.text2sparql.text2sparql_model import Text2SparqlModel
from nemo.collections.nlp.models.text_classification import TextClassificationModel
from nemo.collections.nlp.models.text_normalization_as_tagging import ThutmoseTaggerModel
from nemo.collections.nlp.models.token_classification import (
    PunctuationCapitalizationLexicalAudioModel,
    PunctuationCapitalizationModel,
    TokenClassificationModel,
)
from nemo.collections.nlp.models.zero_shot_intent_recognition import ZeroShotIntentModel
