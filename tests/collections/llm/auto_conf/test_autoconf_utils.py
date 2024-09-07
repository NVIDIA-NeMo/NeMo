from nemo.collections.llm.tools.auto_configurator.core.base_config import _estimate_training_time, calculate_model_size


class TestUtils:
    def test_calculate_model_size(self):
        # GPT
        model_size = calculate_model_size(
            8,
            7,
            None,
            140,
            300,
            "gpt3",
        )
        assert model_size == 0.28, f"expected model_size is 0.28 but got {model_size}."

        # Llama
        model_size = calculate_model_size(
            128,
            30,
            None,
            100,
            3000,
            "llama",
        )
        assert model_size == 1.38, f"expected model_size is 1.38 but got {model_size}."

        # Mixtral
        model_size = calculate_model_size(
            256,
            20,
            None,
            140,
            600,
            "mixtral",
        )
        assert model_size == 12.9, f"expected model_size is 12.9 but got {model_size}."

        # Mistral
        model_size = calculate_model_size(
            1028,
            30,
            None,
            240,
            100,
            "mistral",
        )
        assert model_size == 799.37, f"expected model_size is 799.37 but got {model_size}."

        # Gemma
        model_size = calculate_model_size(
            512,
            30,
            None,
            240,
            100,
            "gemma",
        )
        assert model_size == 398.13, f"expected model_size is 398.13 but got {model_size}."

        # Nemotron
        model_size = calculate_model_size(
            256,
            15,
            None,
            240,
            120,
            "gemma",
        )
        assert model_size == 82.94, f"expected model_size is 82.94 but got {model_size}."

    def test_calculate_train_time(self):
        # GPT
        train_time = _estimate_training_time(
            175,
            1024,
            140,
            300,
            "gpt3",
        )
        assert train_time == 33.91, f"expected train_time is 33.91 but got {train_time}."

        # Llama
        train_time = _estimate_training_time(
            35,
            512,
            60,
            3000,
            "llama",
        )
        assert train_time == 316.48, f"expected train_time is 316.48 but got {train_time}."

        # Mixtral
        train_time = _estimate_training_time(
            0.8,
            128,
            140,
            1000,
            "mixtral",
        )
        assert train_time == 4.13, f"expected train_time is 4.13 but got {train_time}."

        # Mistral
        train_time = _estimate_training_time(
            11,
            24,
            60,
            250,
            "mistral",
        )
        assert train_time == 176.83, f"expected train_time is 176.83 but got {train_time}."

        # Gemma
        train_time = _estimate_training_time(
            7,
            8,
            55,
            100,
            "gemma",
        )
        assert train_time == 147.31, f"expected train_time is 147.31 but got {train_time}."

        # Nemotron
        train_time = _estimate_training_time(
            14,
            12,
            11,
            55,
            "nemotron",
        )
        assert train_time == 540.12, f"expected train_time is 540.12 but got {train_time}."
