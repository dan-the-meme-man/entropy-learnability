import torch
from torch import nn

class ReturnObject:
    
    def __init__(self, **kwargs):
        
        """
            Initialize the return object.
            
            Args:
                kwargs: `dict` - the key-value pairs to store.
        """
        
        for key, value in kwargs.items():
            setattr(self, key, value)

class FFNNLMHeadModel(nn.Module):
    
    def __init__(self, **kwargs):
        
        """
            Initialize the FFNN model.
            
            Args:
                vocab_size: `int` - the size of the vocabulary.
                n_positions: `int` - the number of positions.
                n_embd: `int` - the embedding dimension.
                n_layer: `int` - the number of layers.
                n_head: `int` - the number of heads.
                resid_pdrop: `float` - the dropout probability for residual connections.
                embd_pdrop: `float` - the dropout probability for embeddings.
                lstm_pdrop: `float` - the dropout probability for FFNN layers.
                summary_first_dropout: `float` - the dropout probability for the first summary layer.
                bos_token_id: `int` - the beginning of sentence token ID.
                eos_token_id: `int` - the end of sentence token ID.
        """
        
        super(FFNNLMHeadModel, self).__init__()
        k = 6.11
        
        self.vocab_size = kwargs['vocab_size']
        self.n_positions = kwargs['n_positions']
        self.n_embd = kwargs['n_embd']
        self.n_layer = kwargs['n_layer']
        self.embd_pdrop = kwargs['embd_pdrop']
        self.ffnn_pdrop = kwargs['attn_pdrop']
        
        self.embeddings = nn.Embedding(self.vocab_size, self.n_embd)
        self.embeddings_dropout = nn.Dropout(self.embd_pdrop)
        self.ffnn = nn.ModuleList(
            [
                nn.Linear(self.n_embd, int(self.n_embd * k)),
                nn.ReLU(),
                nn.Dropout(self.ffnn_pdrop)
            ] +
            ([
                nn.Linear(int(self.n_embd * k), int(self.n_embd * k)),
                nn.ReLU(),
                nn.Dropout(self.ffnn_pdrop)
            ] * (self.n_layer - 2)) +
            [
                nn.Linear(int(self.n_embd * k), self.n_embd),
                nn.ReLU(),
                nn.Dropout(self.ffnn_pdrop)
            ]
        )
        self.output = nn.Linear(self.n_embd, self.vocab_size)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> torch.Tensor:
        
        """
            Forward pass of the model.
            
            Args:
                input_ids: `torch.Tensor` - the input tensor.
                attention_mask: `torch.Tensor` - the attention mask tensor.
                
            Returns:
                `torch.Tensor` - the output tensor.
        """
        
        # Embed the input tensor.
        x = self.embeddings(input_ids)
        x = self.embeddings_dropout(x)
        
        # Pass the embeddings through the FFNN.
        for layer in self.ffnn:
            x = layer(x)
        
        # Get the logits.
        x = self.output(x)
        
        outputs = {
            'logits': x
        }
        
        if labels is not None:
            labels = labels.to(x.device)
            shift_logits = x[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            outputs['loss'] = loss
        
        return ReturnObject(**outputs)
