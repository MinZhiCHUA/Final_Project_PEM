import re


def normalize_digits(attribute_value: str):
    try:
        output = f"{float(attribute_value):g}"
        output = output if output != "0" else None
        return output
    except ValueError:
        return None


def normalize_text(attribute_value: str):
    try:
        output = attribute_value.strip()
        output = output if output and output not in {"-"} else None
        return output
    except ValueError:
        return None


def find_offsets_digits(text: str, digit_attribute: str):
    attribute_escaped = re.escape(digit_attribute)
    attribute_comma_escaped = re.escape(digit_attribute.replace(".", ","))
    attribute_pattern = "(%s)|(%s)" % (attribute_escaped, attribute_comma_escaped)

    pattern = r"(^|(?<=[^0-9.,]))(%s)((?=[^0-9.,])|$)" % attribute_pattern

    matches = list(re.finditer(pattern, text))

    entities = [(m.group(), m.start(), m.end()) for m in matches]

    return entities


def find_offsets_text(text: str, text_attribute: str):
    text_attribute = re.escape(text_attribute)
    pattern = r"\b(%s)\b" % text_attribute

    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    entities = [(m.group(), m.start(), m.end()) for m in matches]

    return entities


def preprocess_function(question, context, answer_offsets, tokenizer):
    inputs = tokenizer(
        question,
        context,
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs["offset_mapping"]

    if len(answer_offsets):
        start_char, end_char = answer_offsets[0]

        sequence_ids = inputs.sequence_ids(0)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if (
            offset_mapping[context_start][0] > end_char
            or offset_mapping[context_end][1] < start_char
        ):
            start_position = 0
            end_position = 0
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset_mapping[idx][0] <= start_char:
                idx += 1
            start_position = idx - 1

            idx = context_end
            while idx >= context_start and offset_mapping[idx][1] >= end_char:
                idx -= 1
            end_position = idx + 1

        inputs["start_positions"] = [start_position]
        inputs["end_positions"] = [end_position]

    else:
        inputs["start_positions"] = [0]
        inputs["end_positions"] = [0]

    del inputs["offset_mapping"]

    return inputs
