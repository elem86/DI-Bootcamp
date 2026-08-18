# Project: Course Glossary

A tiny command-line tool that looks up definitions for terms from the course.
It is deliberately small — it exists so you can practice using Claude Code.

## How to run the check

```
python test_glossary.py
```

All checks should pass. **One check currently fails on purpose** — the
case-insensitive lookup — and that is the change the tutorial walks you through.

## How to run the tool

```
python glossary.py token
```

## House rules for changes

Claude Code reads this file automatically and follows it. These rules keep our
practice changes safe and easy to review:

- Keep every change small and focused on exactly what was asked.
- Only edit the file we're discussing — do not touch unrelated files.
- Show me the diff before applying an edit, and explain it in one line.
- After a change, run `python test_glossary.py` so we can watch it pass.
