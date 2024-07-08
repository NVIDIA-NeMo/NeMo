# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from nemo.deploy.nlp import NemoQueryLLM


class TritonSettings(BaseSettings):
    _triton_service_port: int
    _triton_service_ip: str
    _triton_request_timeout: str

    def __init__(self):
        super(TritonSettings, self).__init__()
        try:
            with open(os.path.join(Path.cwd(), 'nemo/deploy/service/config.json')) as config:
                config_json = json.load(config)
                self._triton_service_port = config_json["triton_service_port"]
                self._triton_service_ip = config_json["triton_service_ip"]
                self._triton_request_timeout = config_json["triton_request_timeout"]
        except Exception as error:
            print("An exception occurred:", error)
            return

    @property
    def triton_service_port(self):
        return self._triton_service_port

    @property
    def triton_service_ip(self):
        return self._triton_service_ip

    @property
    def triton_request_timeout(self):
        return self._triton_request_timeout


app = FastAPI()
triton_settings = TritonSettings()


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.0
    n: int = 1
    stream: bool = False
    stop: str | None = None
    frequency_penalty: float = 1.0


@app.post("/v1/completions/")
def completions_v1(request: CompletionRequest):
    try:
        url = triton_settings.triton_service_ip + ":" + str(triton_settings.triton_service_port)
        nq = NemoQueryLLM(url=url, model_name=request.model)
        output = nq.query_llm(
            prompts=[request.prompt],
            max_output_len=request.max_tokens,
            top_k=request.n,
            top_p=request.top_p,
            temperature=request.temperature,
            init_timeout=triton_settings.triton_request_timeout,
        )
        return {
            "output": output[0][0],
        }
    except Exception as error:
        print("An exception occurred:", error)
        return {"error": "An exception occurred"}
