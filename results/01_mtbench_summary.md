# Experiment 01: MT-Bench human judgments (English)

Dataset `lmsys/mt_bench_human_judgments`, split `human`: 3355 rows, 80 questions, 2396 distinct items (question x model pair x turn).

## Label distribution

| winner | count | % |
|---|---|---|
| model_a | 1293 | 38.5 |
| model_b | 1282 | 38.2 |
| tie | 780 | 23.2 |

## Human-human agreement (raw, not kappa)

| labels used | items with 2+ judges | annotator pairs | agreement | chance level |
|---|---|---|---|---|
| with ties | 761 | 1199 | 0.646 | 0.351 |
| ties removed | 536 | 779 | 0.823 | 0.5 |

Chance level = agreement expected if annotators labelled independently with the same label shares. Agreement only means something relative to it.

## Judges per item

{'1': 1635, '2': 599, '3': 132, '4': 24, '5': 6}  (number of humans -> number of items)

## Annotators (rows per judge id)

{'expert_24': 205, 'author_4': 204, 'author_0': 180, 'expert_0': 144, 'author_3': 106, 'expert_9': 103, 'author_2': 102, 'expert_27': 48, 'expert_50': 48, 'expert_47': 48, 'expert_55': 48, 'expert_40': 46, 'expert_26': 46, 'expert_25': 46, 'author_5': 45, 'expert_36': 45, 'expert_31': 44, 'expert_2': 44, 'expert_1': 44, 'expert_12': 44, 'expert_23': 44, 'expert_42': 44, 'expert_8': 44, 'expert_35': 44, 'expert_32': 44, 'expert_6': 44, 'expert_7': 44, 'expert_49': 44, 'expert_29': 44, 'author_6': 44, 'expert_41': 44, 'expert_38': 44, 'expert_53': 44, 'expert_28': 43, 'expert_51': 43, 'expert_22': 43, 'expert_15': 43, 'expert_13': 43, 'expert_14': 43, 'expert_17': 42, 'expert_37': 42, 'expert_45': 42, 'expert_11': 42, 'expert_43': 42, 'expert_10': 42, 'expert_46': 42, 'expert_20': 42, 'expert_57': 42, 'expert_21': 42, 'expert_34': 42, 'expert_16': 42, 'expert_56': 42, 'expert_19': 42, 'expert_5': 42, 'expert_4': 42, 'expert_39': 42, 'expert_44': 42, 'expert_48': 41, 'expert_3': 40, 'expert_33': 12, 'expert_18': 10, 'expert_54': 10, 'expert_30': 6, 'author_1': 6, 'expert_52': 4}

## Example rows (seed 0)

- q159 turn 1: gpt-3.5-turbo vs claude-v1 -> **model_b** (by author_0). Prompt: 'What are some business etiquette norms when doing business in Japan?'
- q123 turn 1: llama-13b vs gpt-3.5-turbo -> **model_b** (by author_4). Prompt: 'Write a simple website in HTML. When a user clicks the button, it shows a random joke from a list of 4 jokes.'
- q160 turn 1: llama-13b vs alpaca-13b -> **model_a** (by expert_22). Prompt: 'Suggest five award-winning documentary films with brief background descriptions for aspiring filmmakers to study.'
