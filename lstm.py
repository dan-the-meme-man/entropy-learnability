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

class LSTMLMHeadModel(nn.Module):
    
    def __init__(self, **kwargs):
        
        """
            Initialize the LSTM model.
            
            Args:
                vocab_size: `int` - the size of the vocabulary.
                n_positions: `int` - the number of positions.
                n_embd: `int` - the embedding dimension.
                n_layer: `int` - the number of layers.
                n_head: `int` - the number of heads.
                resid_pdrop: `float` - the dropout probability for residual connections.
                embd_pdrop: `float` - the dropout probability for embeddings.
                lstm_pdrop: `float` - the dropout probability for LSTM layers.
                summary_first_dropout: `float` - the dropout probability for the first summary layer.
                bos_token_id: `int` - the beginning of sentence token ID.
                eos_token_id: `int` - the end of sentence token ID.
        """
        
        super(LSTMLMHeadModel, self).__init__()
        
        self.vocab_size = kwargs['vocab_size'] + 3
        self.n_positions = kwargs['n_positions']
        self.n_embd = kwargs['n_embd']
        self.n_layer = kwargs['n_layer']
        self.embd_pdrop = kwargs['embd_pdrop']
        self.lstm_pdrop = kwargs['attn_pdrop']
        
        self.embeddings = nn.Embedding(
            self.vocab_size,
            self.n_embd,
            padding_idx=kwargs['pad_token_id']
        )
        self.embeddings_dropout = nn.Dropout(self.embd_pdrop)
        self.lstm = nn.LSTM(
            self.n_embd,
            int(self.n_embd * 1.266),
            self.n_layer,
            dropout=self.lstm_pdrop,
            batch_first=True
        )
        self.output = nn.Linear(int(self.n_embd * 1.266), self.vocab_size)
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=kwargs['pad_token_id'])
        
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
        
        # Pass the embeddings through the LSTM.
        x, h = self.lstm(x)
        
        # Get the logits.
        x = self.output(x)
        
        outputs = {
            'logits': x,
            'hidden_states': h
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
