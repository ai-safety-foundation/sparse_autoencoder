# How do I get past the linting checks?

The first thing you should do is install the visual studio code [Ruff
plugin](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff). This plugin is
going to solve the vast majority of issues you may come across while working on this project. Most
common issues the linter finds will be automatically fixed when saving files with this plugin.

Some common issues caught by the lint checks cannot be automatically fixed. e.g. unknown types from
an outside package, print statements, commented out lines of code, etc... When encountering some of
these issues, there are a few options.

If you are in the middle of working on something, and you simply want to commit your changes in
order to finish what you are doing later, you can edit the file `pyproject.toml` to allow you to
commit your files while temporarily ignoring lint. To do this you want to add an entry to the block
`tool.ruff.lint.per-file-ignores` with your file name as the key, and a list of any lint codes you want to
ignore within those files. Please note that you most likely will need to remove these rules before
submitting a pull request. In almost all cases there is a more permanent solution to skipping lint,
and adding rules to your working files should only be a temporary solution to allow the lint to get
out of your way while developing a work in progress. There are situations where lint needs to be
skipped in this way, but those are few and far between. Most of those situations have also already
been addressed.

In cases where you may need a more permanent bypass of lint checks, you can create case exceptions
for individual lines see [docs for examples](https://docs.astral.sh/ruff/tutorial/#ignoring-errors).
Common cases where this is necessary is when a dependency has poor typing, and the linter complains
about references to those components in your code. More often than not, it is not necessary to skip
lines of files, and this should only be done as a last resort.
