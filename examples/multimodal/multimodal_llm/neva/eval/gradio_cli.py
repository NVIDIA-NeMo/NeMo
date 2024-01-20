# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import base64

import requests

# URL of the Gradio server
url = 'http://localhost:8890/api/predict/'

# Prepare the text data
text_data = '<image>Describe this image please.'

# Prepare the image data
with open("/path/to/images/001.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Data to send
data = {'data': [text_data, encoded_string]}

# Sending a POST request to the Gradio server
response = requests.post(url, json=data)

# Checking if the request was successful
if response.status_code == 200:
    # Parsing the response
    response_data = response.json()
    print("Response from server:", response_data)
else:
    print("Failed to get a response from the server, status code:", response.status_code)
