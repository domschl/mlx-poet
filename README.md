## MLX Poet

WARNING: Early test version. Missing features.

Initial code with training using MLX and ml-indie-tools as data source.

MLX Poet uses your own data to train a transformer model that can generate text. It also uses its own tokenizer, so that the tokens are optimally adjusted to your own data sources which allows support for arbitrary languages without being focused on English.

Data sources that can be used are:

- Books that are downloaded from Project Gutenberg using searches for titles or authors.
- Local notes in TXT or MD format.
- If you are using Calibre for ebooks, it can load TXT formated books from your Calibre library. Use Calibre to convert your ebooks to TXT format.

Using a carefully curated dataset is significantly more efficient and yields far better results than using random internet data.

## Installation

Requires Python 3.9 or later, `mlx`, `numpy`, `matplotlib`, `ml-indie-tools`.

## Configuration

Start `jupyter lab` and open `mlx_poet.ipynb`. Follow the instructions in the notebook.

Things to adjust:

- in `1. Project configuration`, adatapt `project_name`. The `use_preprocessed_data` to `False` only on first run. This will invoke the tokenizer and afterwards save the tokenized data to disk. On subsequent runs, set it to `True` to load the tokenized data from disk. The first tokenizer run can be very slow, depending on the size of the dataset.
- in `2.1 Text data from Project Gutenberg`, adjust the `search_term` to the title or author of the book you want to download. By default a maximum of 40 books are downloaded. Consult the comments on how to tackle the download of more books.
- Create a file `additional_texts.json` in the root directory of the project (next to `mlx_poet.ipynb`). Sample format is:

```json
{
  "local_texts": ["/some/directory/that/contains/texts"],
  "calibre": "/home/myuser/Calibre Library"
}
```
- `local_texts` is a list of directories that contain text files. The files can be in TXT, MD, ORG, or PY format.
- `calibre` is the path to your Calibre library. All books that are available in TXT format in your Calibre library will be used.
- in `3. Model metadata` adjust the number of transformer layers (`attn_layers`), the embedding size (`emb`) that is also used as sequence length by default.

## Not implemented yet:

- Model state is not saved or checkpointed yet.

