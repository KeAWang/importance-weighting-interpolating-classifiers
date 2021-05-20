import torch
from transformers import DistilBertForSequenceClassification
from omegaconf import DictConfig


class DistilBertNet(torch.nn.Module):
    def __init__(
        self, model_name: str, hf_config: DictConfig, input_size: int, output_size: int,
    ):
        """Note that if output_size == 1, this will be a regression model. See """
        super().__init__()
        # TODO: assert input_size is same as what hf_model expects
        self.model_name = model_name
        self.output_size = output_size
        self.hf_model = DistilBertForSequenceClassification.from_pretrained(
            model_name, **hf_config
        )
        # hidden_size = (
        #    self.hf_model.config.hidden_size
        # )  # config.hidden_size is same as config.dim; see https://huggingface.co/transformers/_modules/transformers/models/distilbert/configuration_distilbert.html#DistilBertConfig

    def forward(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        output = self.hf_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        logits = output.logits
        return logits

    @property
    def linear_output(self):
        return self.hf_model.classifier
