self.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # _load_in_4bit=True,
    # _load_in_8bit=True,
    # quant_method='your_quant_method',
) 