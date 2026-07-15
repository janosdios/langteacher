# Localization

CLI-facing strings (prompts, `--help` banners, startup/status messages) are translated
via Python's `gettext`. This does **not** cover what the tutor LLM says out loud — that's
controlled by `TARGET_LANGUAGE`/`NATIVE_LANGUAGE` in [`llm_engine.py`](../llm_engine.py),
not by this system.

## How it's wired

- One gettext **domain** per module that has translatable strings — `main`,
  `llm_engine`, `stt_engine`, `tts_engine`, `languages`. Each domain has its own
  `_()` function, built from `gettext.translation(domain, localedir=..., languages=[...],
  fallback=True)` near the top of that module.
- `locale/<domain>.pot` is the **template**: the master list of translatable strings for
  that domain, extracted from source. It has no translations in it (`msgstr` is always
  empty) and is not read at runtime — it only exists to generate/update `.po` files.
- `locale/<lang>/LC_MESSAGES/<domain>.po` holds the actual translations for one
  language/domain pair, in plain text. This is what you hand-edit.
- `locale/<lang>/LC_MESSAGES/<domain>.mo` is the compiled binary form of the `.po` file
  — this is what `gettext.translation()` actually loads at runtime. **Every `.po` change
  needs recompiling to `.mo` or it won't take effect.**
- `fallback=True` means a missing catalog, or a missing/empty `msgstr` inside one, just
  falls back to the original English string instead of crashing — so partial or missing
  translations are never fatal, just incomplete.
- The active language is picked via the `UI_LANGUAGE` env var or the `--ui-lang <code>`
  CLI flag (e.g. `hu`, `de`), read at startup by each module and (for `main.py`)
  propagated to the others via `set_ui_language()`.
- `hu` (Hungarian) is currently the only translated locale.

Requires the `gettext` command-line tools (`xgettext`, `msgmerge`, `msgfmt`) — on macOS,
`brew install gettext && brew link gettext --force`; on Debian/Raspberry Pi OS,
`apt install gettext`.

---

## 1. Change an existing string's translation in an existing language

Say you want to fix the Hungarian wording of a `main.py` message.

1. Open `locale/hu/LC_MESSAGES/main.po` and find the entry — each one is a comment
   (`#: main.py:<line>`) followed by `msgid "<original English>"` and
   `msgstr "<translation>"`. Edit the `msgstr`.
2. Recompile that one file to `.mo`:
   ```bash
   msgfmt --check --statistics -o locale/hu/LC_MESSAGES/main.mo locale/hu/LC_MESSAGES/main.po
   ```
   `--check` catches malformed strings (mismatched `{placeholders}`, bad escapes) before
   they'd silently break formatting at runtime; `--statistics` confirms how many messages
   compiled.
3. Verify: `UI_LANGUAGE=hu python3 main.py --help` (or run the specific flow that prints
   the string) and confirm the new wording shows up.

No need to touch the `.pot` file — you're only changing a translation, not the set of
translatable strings.

---

## 2. Add a new language

Say you want to add German (`de`) support for the `main` domain (repeat per domain —
`llm_engine`, `stt_engine`, `tts_engine`, `languages` — for full coverage).

1. Create the directory and seed a `.po` from the domain's template:
   ```bash
   mkdir -p locale/de/LC_MESSAGES
   msginit --no-translator --locale=de --input=locale/main.pot \
     --output-file=locale/de/LC_MESSAGES/main.po
   ```
   `msginit` fills in the `.po` header (language, plural forms) for you; every `msgstr`
   still starts out empty.
2. Translate: open `locale/de/LC_MESSAGES/main.po` and fill in every `msgstr`. Keep
   `{placeholder}` names and `\n` escapes exactly as they appear in the `msgid` — they're
   substituted at runtime (see the `#, python-brace-format` comment above such entries).
3. Compile:
   ```bash
   msgfmt --check --statistics -o locale/de/LC_MESSAGES/main.mo locale/de/LC_MESSAGES/main.po
   ```
4. Repeat steps 1–3 for the other domains (`llm_engine`, `stt_engine`, `tts_engine`,
   `languages`) using their own `.pot` files, so the language is fully covered rather
   than just `main.py`'s strings.
5. Verify: `UI_LANGUAGE=de python3 main.py --help`, and spot-check a couple of the other
   modules' own `--help`/prompts too.

Untranslated entries aren't an error (`fallback=True` shows the English original), so you
can ship a partial translation and fill in the rest incrementally — just don't forget to
recompile the `.mo` after each edit.

---

## 3. Add a new translatable string in source code

Say you're adding a new user-facing message in, e.g., `main.py`.

1. Wrap the literal in that module's `_()`, same as every other string in the file:
   ```python
   print(_("Some new message: {detail}").format(detail=value))
   ```
   (Every module that has an `_()` already defines it near the top via
   `gettext.translation(...).gettext` — don't add a new one.)
2. Regenerate that module's `.pot` so the new string is in the template:
   ```bash
   xgettext --language=Python --keyword=_ --from-code=UTF-8 --package-name=main \
     -o locale/main.pot main.py
   ```
   (swap `main`/`main.py` for whichever domain/module you edited).
3. Merge the updated template into every existing language's `.po` for that domain —
   this adds the new (empty) entry and updates `#:` line-number comments, without
   touching translations that haven't changed:
   ```bash
   msgmerge --update --backup=off locale/hu/LC_MESSAGES/main.po locale/main.pot
   ```
   Run once per language directory that has this domain. `msgmerge` will try to guess a
   translation for the new string from similar existing ones and mark the guess
   `#, fuzzy` — always review (or clear) fuzzy entries rather than trusting them blindly.
4. Fill in the new `msgstr` (and resolve any `#, fuzzy` marks) in each language's `.po`.
   To find what's left to do:
   ```bash
   msgattrib --untranslated locale/hu/LC_MESSAGES/main.po   # empty msgstr
   msgattrib --only-fuzzy locale/hu/LC_MESSAGES/main.po     # needs review
   ```
5. Recompile each language's `.mo`:
   ```bash
   msgfmt --check --statistics -o locale/hu/LC_MESSAGES/main.mo locale/hu/LC_MESSAGES/main.po
   ```
6. Verify with `UI_LANGUAGE=hu` (and default/English, i.e. no `UI_LANGUAGE` set) that the
   new string prints correctly in both.

It's fine to merge and leave the new string untranslated for a while (English shows via
fallback) — just don't skip step 5, since an unrecompiled `.po` edit has no effect at all.
