# Contributing

## Setup

This project uses [Poetry](https://python-poetry.org) for dependency management, and
[PoeThePoet](https://poethepoet.natn.io/installation.html) for scripts. After checking out the repo,
we recommend setting poetry's config to create the `.venv` in the root directory (note this is a
global setting) and then installing with the dev and demos dependencies.

```shell
poetry config virtualenvs.in-project true
poetry install --with dev,demos
```

If you are using VSCode we highly recommend installing the recommended extensions as well (it will
prompt you to do this when you checkout the repo).

## Checks

For a full list of available commands (e.g. `test` or `typecheck`), run this in your terminal
(assumes the venv is active already).

```shell
poe
```

## Documentation

Please make sure to add thorough documentation for any features you add. You should do this directly
in the docstring, and this will then automatically generate the API docs when merged into `main`.
They will also be automatically checked with [pytest](https://docs.pytest.org/) (via
[doctest](https://docs.python.org/3/library/doctest.html)).

If you want to view your documentation changes, run `poe docs-hot-reload`. This will give you
hot-reloading docs (they change in real time as you edit docstrings).

### Docstring Style Guide

We follow the [Google Python Docstring Style](https://google.github.io/styleguide/pyguide.html) for
writing docstrings. Some important details below:

#### Sections and Order

You should follow this order:

```python
"""Title In Title Case.

A description of what the function/class does, including as much detail as is necessary to fully understand it.

Warning:

Any warnings to the user (e.g. common pitfalls).

Examples:

Include any examples here. They will be checked with doctest.

  >>> print(1 + 2)
  3

Args:
    param_without_type_signature:
        Each description should be indented once more.
    param_2:
        Another example parameter.

Returns:
    Returns description without type signature.

Raises:
    Information about the error it may raise (if any).
"""
```

#### LaTeX support

You can use LaTeX, inside `$$` for blocks or `$` for inline

```markdown
Some text $(a + b)^2 = a^2 + 2ab + b^2$
```

```markdown
Some text:

$$
y    & = & ax^2 + bx + c \\
f(x) & = & x^2 + 2xy + y^2
$$
```

#### Markup

- Italics - `*text*`
- Bold - `**text**`
- Code - ` ``code`` `
- List items - `*item`
- Numbered items - `1. Item`
- Quotes - indent one level
- External links = ``` `Link text <https://domain.invalid/>` ```
