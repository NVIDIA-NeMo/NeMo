Datasets
========

Input data should be provided in line delimited JSON format as below:
	
.. code-block:: bash

  {"audio_filepath": "/path/to/abcd.wav", "offset": 0, "duration": 10.1, "text": "{'scenario': 'Calendar', 'action': 'Create_entry', 'entities': [{'type': 'event_name', 'filler': 'brunch'}, {'type': 'date', 'filler': 'Saturday'}, {'type': 'timeofday', 'filler': 'morning'}, {'type': 'person', 'filler': 'Aronson'}]}"}

The semantics annotation is a Python dictionary flattened as a string, and indexed by the "text" key in the manifest. For a semantics annotation, there are three mandatory keys: "scenario", "action" and "entities". The values for "scenario" and "action" are strings, where the value for "entities" is a Python list of dictionary. Each item in "entities" is also a Python dictionary, with two keys "type" (entity slot) and "filler" (slot filler).
