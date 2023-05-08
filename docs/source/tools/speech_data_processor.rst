Speech Data Processor
========================

Speech Data Processor (SDP) is a toolkit to make it easy to:
  1. write code to process a new dataset, minimizing the amount of boilerplate code required.
  2. share the steps for processing a speech dataset.

SDP is hosted here: https://github.com/NVIDIA/NeMo-speech-data-processor. 

SDP's philosophy is to represent processing operations as 'processor' classes, which take in a path to a NeMo-style data manifest as input (or a path to the raw data directory if you do not have a NeMo-style manifest to start with), apply some processing to it, and then save the output manifest file.

You specifiy which processors you want to run using a YAML config file. Many common processing operations are provided, and it is easy to add your own. If you do not need to add your own processors, then all that is needed to process a new dataset is to write a single YAML file containing the parameters needed to process your dataset.

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.17.0/sdp_overview_diagram.png
  :alt: Overview diagram of Speech Data Processor

Overview of how SDP processes a dataset
---------------------------------------

1. You call the ``main.py`` script, passing in a YAML config file, possibly with some overrides.
2. ``main.py`` script calls ``run_processors.py``, passing in your config.
3. ``run_processors.py`` does the following:

   a. picks out the processors that you specified to be run (you can specify a subset of the processors in the config override, e.g. to avoid re-running time-consuming steps).
   b. if some of the processors have not had "output_manifest_file" or "input_manfiest_file" entries specified, SDP will automatically create temporary files for those.
   c. instantiates the processor classes using ``hydra.utils.instantiate``
   d. runs the run-time processor tests by calling the ``processor.test()`` method (more details about testing :ref:`here<SDP Tests>`).
   e. runs the processing method (``processor.process()``) of each processor in order.
  

Layout of config YAML files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The YAML config file for processing a dataset must contain a key ``processors``, the value of which is a list. Each item in that list is expected to be a dictionary specifying a processor class, i.e. it must have a key ``_target_``, the value of which is a path to a "processor" class, and the remaining keys must be the kwargs necessary to instantiate that class with ``hydra.utils.instantiate()`` (c.f. https://hydra.cc/docs/advanced/instantiate_objects/overview/).

SDP will run the processors specified in the ``processors`` list in the config file. It will also check for a ``processors_to_run`` key in the config file, which can be either the string ``"all"``, or any Python "slice" object like ``3:4``, ``2:`` etc. (if there is no ``processors_to_run`` key, then all of the processors will be run).

.. note:: 
    SDP will run the processors in the order in which they are listed in the config YAML file. Make sure to list the processors in an order which makes sense, e.g. create an initial manifest first; make sure to run asr inference before doing any processing which looks at ``pred_text`` fields in the manifest.

Processor classes
-----------------

**BaseProcessor**
~~~~~~~~~~~~~~~~~

All processor classes inherit from the ``BaseProcessor`` class. This is a simple abstract class which has 2 empty methods: ``process()`` and ``test()``. 
These serve to remind us that SDP essentially just runs ``test()`` on all processors, and then ``process()`` on all processors (more details about testing :ref:`here<SDP Tests>`).

``ASRInference`` is a child class of ``BaseProcessor``. It has a simple ``process()`` method which runs transcription on every utterance in the input_manifest.

``WriteManifest`` is also a child class of ``BaseProcessor``. It has a simple ``process()`` method which saves a copy of the input manifest containing only the fields specified in ``fields_to_save``.

**BaseParallelProcessor**
~~~~~~~~~~~~~~~~~~~~~~~~~
``BaseParallelProcessor`` inherits from the ``BaseProcessor`` class. Within the ``BaseParallelProcessor.process()`` method, it calls other methods and functions, which allow it to do more complex processing. 
Most importantly, it calls its ``BaseParallelProcessor.process_dataset_entry(data_entry)`` method on every utterance in the manifest, and it does this in parallel, allowing for more efficient processing.

What is a **DataEntry**?
~~~~~~~~~~~~~~~~~~~~~~~~
As mentioned above, ``BaseParallelProcessor.process_dataset_entry(data_entry)`` is called on a variable called ``data_entry`` which represents an utterance in our dataset.
Most often, ``data_entry`` will be a dictionary containing items which represent the JSON manifest entry. 
Sometimes, such as in ``CreateInitialManifestMLS``, it will be a string containing a line for that utterance from the original raw MLS transcript.

``BaseParallelProcessor.process_dataset_entry`` will process ``data_entry`` and output a ``DataEntry`` object. 

The ``DataEntry`` class is a dataclass which contains 2 attributes:

1. ``data`` is an Optional dictionary containing items which represent the JSON manifest entry. ``data`` can also be ``None``. If a ``.process_dataset_entry(data_entry)`` method returns a ``DataEntry`` class where ``data is None``, then that utterance will be dropped from the output manifest.
2. ``metrics``, which can be of any type, and are ``None`` by default. This variable is used by some variables to record summary statistics about the changes made to the dataset, these metrics are aggregated and can be displayed once every utterance has been processed by the processor.

What happens in **BaseParallelProcessor.process()**?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We outline the ``BaseParallelProcessor.process()`` method below:

.. raw:: html

    <div align="center">
      <img src="https://mermaid.ink/img/pako:eNplUl1r6zAM_SvCFy4pbL3vvaVwu-59sL0tl6LESmqIP7DkjWzsv89O0rVjzosiHR8dHetdtV6T2qg-YjjB0-Fv7SAfTs2cqdWjUGAwDrYiuz0yPWDEYaDhIfqWmH1chzmqVts_GQOW5OR1rWaqcv4916pcZxq6jKaAkRb0tok7IBtkXO5BM4KmDtMgUIotOmgIEpMG8VOK1v0atH91g0cNEV9BoyBgEm9RTJvljbX6D7e3O9hfVOyvVURCfbToTEcs11pKocwbksC5PnWFyhB00VvIE7wYnxiWwY3rgbNNqwlnOpATRQLD4B2dhdxdhNx9t2PiOJYRmORITuJYlb85XEydFGDDErGVL4tn6gNcuA-Zm_GFwCf5McJvwL6P1KNQoYim5SlfTY7-At9BEmHQ0YdAenVucH_hv7_W3hmHg3mj40JWXYudX8lwGHD86rb4d7YtN6hd-Qo1Oa1ulKVo0ei8k-8lXatsps0ubnK47EVZrY8MLQ_-OLpWbSQmulEpZNvoYDDvrlWbDgemj0-10vX9" height=100% />
    </div>


**ModifyManifestTextProcessor**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ModifyManifestTextProcessor`` inherits from the ``BaseParallelProcessor`` class. 

The ``ModifyManifestTextProcessor`` constructor takes in the following arguments: 
* ``text_key`` (string) and ``pred_text_key`` (string): these parameters specify which keys in ``data_entry.data`` will be used for processing. (default: ``text_key="text"``, ``pred_text_key="pred_text"``, ie. by default the processor will refer to and modify the ``"text"`` and/or ``"pred_text"`` attributes of the input manifest).
* ``test_cases`` (optional, list of dicts) - test cases for checking that the processor makes the changes that we are expecting.

``ModifyManifestTextProcessor`` has the following methods: 
* ``ModifyManifestTextProcessor.test()``: this method makes sure that the output from the processor matches the expected output specified in the ``test_cases`` parameter.
* ``ModifyManifestTextProcessor.process_dataset_entry(data_entry)``: this method applies processing to a ``data_entry``. First, spaces are added to the start and end of the 'text' and 'pred_text' entries (if they exist), then the abstract method ``ModifyManifestTextProcessor._process_dataset_entry(data_entry)`` is called. Then, any extra spaces (e.g. two spaces next to each other '  ') are removed from 'text' and 'pred_text' entries.
* ``ModifyManifestTextProcessor._process_dataset_entry(data_entry)``: this is an abstract method which will be over-written by children of ``ModifyManifestTextProcessor``.

How to make your own processor classes
--------------------------------------

We will describe how to make your own processor classes by referring to SDP's existing classes.

Creating an initial manifest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One of the child classes of ``BaseParallelProcessor`` provided in SDP is ``CreateInitialManifestMLS``. It downloads raw MLS data for a specified language, and creates an initial manifest (in the format expected by NeMo) which can be cleaned by subsequent processors.

The ``CreateInitialManifestMLS.prepare()`` method downloads and extracts the raw data.

The ``CreateInitialManifestMLS.read_manifest()`` method reads the lines in the raw MLS transcript file.

The ``CreateInitialManifestMLS.process_dataset_entry()`` method takes in the lines from the raw MLS transcript file, and outputs ``DataEntry`` objects containing entries that will be saved into the manifest (i.e. ``"audio_filepath"``, ``"duration"``, ``"text"``) for each utterance.


A **ModifyManifestTextProcessor** subclass that cleans the reference text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the classes provided in SDP is ``SubRegex``. At initialization, it takes in ``regex_params_list``, a list of dictionaries which must contain the keys ``"pattern"``, ``"repl"``, and, optionally, ``"count"``. These keys will be used to apply regex substitutions using these parameters fed into ``re.sub``. The substitutions will be applied to the data at ``text_key`` (i.e. ``data_entry.data[self.text_key]``). By default, ``text_key="text"``, i.e. the substitutions will be applied to the ``"text"`` attribute of the manifest.

In its ``_process_dataset_entry(data_entry)`` method, the ``SubRegex`` processor does the string to string conversion upon the ``data_entry`` that is input. Its output is a ``data_entry`` with the changes applied to ``data``, and the the metrics of which regex patterns caused a substitution to be made. These metrics will be aggregated over all utterances by the ``BaseParallelProcessor`` class. ``SubRegex`` also has a ``finalize(metrics)`` method which will log information about the aggregated metrics after all of the utterances in the manifest have been processed.

A **ModifyManifestTextProcessor** subclass that drops incorrectly transcribed utterances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the classes provided in SDP is ``DropHighLowCharrate``. At initialization, it takes in ``high_charrate_threshold`` and ``low_charrate_threshold``, for which the utterance will be dropped if it is above or below each value respectively. This is helpful for automatically filtering out incorrectly transcribed utterances.

In its ``_process_dataset_entry(data_entry)`` method it evaluates the character rate of the utterance(by dividing the length of ``data_entry.data[self.text_key]`` by the value of ``data_entry.data["duration"]``). If the character rate is within bounds, it will return the same ``data_entry`` that was input. If the character rate is out of bounds, it will return a ``data_entry`` with ``data=None`` and ``metrics`` which reflect the applied changes.
Similar to the ``SubSubstringToSpace`` class, it has a ``finalize(metrics)`` method which will log information about the aggregated metrics after all of the utterances in the manifest have been processed.

Class diagram
-------------
A diagram of the classes mentioned above is included here. Arrows represent inheritance.

We omit the details of the ``CreateInitialManifestMLS`` class in the diagram in order to save space.


.. raw:: html

    <div align="center">
      <img src="https://mermaid.ink/img/pako:eNqlVMFu2zAM_ZVApw1o8wHBLl17WIEGGOYCuxgQWImOhcqSQdFtM6__PjmSvbhzsgI1fKDI98gnklAvlNcoNkJZCOHGwI6gKd1XCPidvMIQPK2-_L68XB1cQGAt2im0iLwqfty6CgmdwkXATzKMW3CmwsCly5i3uRP2mhAYb51hA3bkbO-Ks6St16baj-h7fOEjxaU7E078onuIf2AybnfvixaGi_yXdUO-_WZ29Z1_vq6BKOoeqh06u5q1oS_dKn6-47Zj2eSUsjIWU8S4E4E2pfj0OR05Rgf7dVbmbVP6RW5L2ALheIx91lPFv5gDRWrgmJglOqb9GKyMA2t-4UzA8fCnusgExmHMH5fNJu8DsKpliPx_1E3JZovSj1XR6iDZywBPZ7inFienWa_Xk7GeEc_MuR-7_sLyEffT9bScu4axSBU7FuZjOt3S4ZTMDJPvwE2SF_Y1Sw2jO7w_7Wy2TZydUeG42sKe52p19EqVfZJrwlB7q1PQ-ueTsQ_IisLEhWiQGjA6PmQHKaXgGhssxSaaGivoLJciQaFjX-ydEpsKbMAL0bWxDua3L3tf_wDMstkP" height=100% />
    </div>

SDP Tests
---------
It is important to make sure that your data processing code has the effect you intend, so SDP has a few different types of tests:

1. Runtime tests

* Before running the specified processors, SDP runs ``processor.test()`` on all specified processors.
* Currently, the only provided processor classes with a test method are subclasses of ``ModifyManifestTextProcessor``.

  * ``ModifyManifestTextProcessor.test()`` runs any ``test_cases`` that were provided in the object constructor.
  * This means you can provided test cases in the YAML config file, and the dataset will only be processed if the test cases pass.
  * This is helpful to (a) make sure that the rules you wrote have the effect you desired, and (b) demonstrate why you wrote those rules.
  * An example of test cases we could include in the YAML config file::

      - _target_: sdp.processors.DropIfRegexMatch
        regex_patterns:
          - "(\\D ){5,20}" # looks for between 4 and 19 characters surrounded by spaces
        test_cases:
          - {input: {text: "some s p a c e d out letters"}, output: null}
          - {input: {text: "normal words only"}, output: {text: "normal words only"}}

2. ``pytest`` tests which can be run locally with ``python -m pytest tests/`` and will be run during the GitHub CI process. There are 2 sub-types:

   a. "End to end" tests (link) which run SDP on a mini version of the raw initial dataset, and make sure the final manifest matches the reference final manifest. 
   b. "Unit tests" for processors and utils (link).
