from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


tokenizer.pad_token = tokenizer.eos_token

# print tokenizer tokens

print(
    f"pad_token = {tokenizer.pad_token} {tokenizer.encode(tokenizer.pad_token)}")
print(
    f"eos_token = {tokenizer.eos_token} {tokenizer.encode(tokenizer.eos_token)}")
print(
    f"bos_token = {tokenizer.bos_token} {tokenizer.encode(tokenizer.bos_token)}")
print(
    f"unk_token = {tokenizer.unk_token} {tokenizer.encode(tokenizer.unk_token)}")
print(
    f"additional_special_tokens = {tokenizer.additional_special_tokens}")
print(
    f"additional_special_tokens_ids = {tokenizer.additional_special_tokens_ids}")


input = "Hello world"
cand = "another hello world"

annotated_input = [f'{tokenizer.bos_token} {input}', f' {cand}']

# encoding1 = tokenizer.encode(annotated_input[0])
# print(f'encoding1: {encoding1}')

# encoding2 = tokenizer(annotated_input)
# print(f'encoding2: {encoding2}')

encoding3 = tokenizer.batch_encode_plus([annotated_input],
                                        padding=True,
                                        truncation=True,  # TODO
                                        max_length=512,
                                        return_tensors='pt',
                                        return_attention_mask=False,
                                        return_token_type_ids=True)
print(f'encoding3: {encoding3}')

# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained(
    "gpt2", pad_token_id=tokenizer.eos_token_id)


with torch.no_grad():
    # Put the model in inference mode.
    model.eval()

    # token_type_ids is 0 for the first string (query) and 1 for the second
    # string (candidate).
    token_type_ids = encoding3.token_type_ids
    print(f'token_type_ids ${token_type_ids.shape}: {token_type_ids}')

    # Perform model inference
    outputs = model(input_ids=encoding3.input_ids)

    # The logits output is a tensor with shape [batch size, token sequence
    # length, vocabulary size]. It is the raw logits before softmax.
    print(f'output.logits.shape: {outputs.logits.shape}')

    # cumsum is a tensor with the same shape as input_ids where each element
    # counts the number of tokens prior that belong to the second string
    # (candidate).
    cumsum = token_type_ids.cumsum(1)
    print(f'cumsum ${cumsum.shape}: {cumsum}')

    # The "window size" is the number of tokens in the candidate to consider
    # when determining the probability it follows the query. A lower number
    # biases the result towards the "surprise" that the first few words are not
    # expected to follow from the query. A higher number increases the context,
    # but it also tends to trend towards "less surprise" as more and more tokens
    # are considered, and the condidate itself provides more and more context
    # for words later in the candidate string. This is a parameter that should
    # probably be experimented with.
    win_size = 8

    # `mask` is a tensor with the same shape as input_ids where each element in
    # the canidate that fits within the window is True, and all other elements
    # are False. This masks out only the tokens whose logits are considered when
    # computing the probability that they follow the query. The pad operation
    # strips the first element, and adds a False at the end. This allows the
    # mask to account for the removal of the bos_token in the padded variable
    # (defined next), as well as mask out the final value where the "next token"
    # is artificially set to the 0th index, and whose log probability is
    # extraneous.
    mask = F.pad((token_type_ids == 1) & (cumsum <= win_size),
                 [-1, 1])
    print(f'mask ${mask.shape}: {mask}')

    # `padded` uses the pad function to remove the first element of the
    # input_ids tensor and add a 0 to the end. The first element is removed as
    # it's artificially an inserted bos_token. The 0 is added to the end to make
    # the tensor the same shape as input_ids.
    padded = F.pad(encoding3['input_ids'], [-1, 1])
    print(f'padded ${padded.shape}: {padded}')

    # `input_ids` adjusts the `padded` tensor by using the indexing operator
    # with the value None, which inserts a 1-element dimension at the end of the
    # tensor (e.g. each value of `padded` becomes its own 1-dimensional tensor).
    # This is necessary in preparation for the `gather` operator.
    input_ids = padded[:, :, None]
    print(f'input_ids ${input_ids.shape}: {input_ids}')

    # `softmax` is a tensor representing the log softmax of the logits output of
    # the language model. For each input token, a vector of log probabilities
    # for each possible output token.
    softmax = outputs['logits'].log_softmax(2)
    print(f'softmax ${softmax.shape}: {softmax}')

    # `gathered` is a tensor with log probabilities for each input token. The
    # gather function selects the index of the softmax vector that corresponds
    # to each input token. This is the log probability that the token appears at
    # that position in the sentence formed by the concatenation of the query and
    # candidate.
    gathered = softmax.gather(2, input_ids)
    print(f'gathered ${gathered.shape}: {gathered}')

    # This removes the 1-dimensional dimension from the gathered tensor. The
    # result is a tensor with the log probabilities for each input token.
    squeezed = gathered.squeeze(2)
    print(f'squeezed ${squeezed.shape}: {squeezed}')

    # This applies the mask to the log probabilities. The mask allows through
    # only the first win_size probabilities corresponding to tokens of the
    # candidate.
    masked = squeezed.masked_fill(~mask, 0)
    print(f'masked ${masked.shape}: {masked}')

    # This sums the log probabilities for the candidates in the masked window.
    summed = masked.sum(1)
    print(f'summed ${summed.shape}: {summed}')

    # This is the average log probability of the candidate over the window size
    # (computed using the mask to handle candidates shorter than the window
    # size).
    scores = summed / mask.sum(1)
    print(f'mask.sum(1) ${mask.sum(1)}')

    # print scores
    print(f'scores: {scores}')
