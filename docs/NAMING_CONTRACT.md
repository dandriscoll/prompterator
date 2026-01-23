# ft — File-Naming Tool Contract

## Synopsis

```
ft config
ft propose <path> <type>
ft ready <path>
ft bundles [dir]
```

## Output

- Plain text to `stdout`
- Exit `0` on success, non-zero on failure

## Operations

### `config`

Print tool configuration in a structured format.

```
$ ft config
prior-types: prompt.txt, prompt.md
source-type: out.txt
feedback-type: mb
```

Fields:
- `prior-types` — Comma-separated list of prior file types
- `source-type` — The source/output file type
- `feedback-type` — The feedback/annotation file type

### `propose <path> <type>`

Propose a new filename derived from `<path>` with the target `<type>`.

- Preserves: index, variation, name
- Changes: extension (per target type)
- Iteration: if input type equals target type, increments variation
- Collision: checks filesystem, increments variation until filename is available

```
$ ft propose 001.prompt.txt out.txt
./001.out.txt

$ ft propose 001.out.txt out.txt
./001a.out.txt

$ ft propose 001a.out.txt out.txt
./001b.out.txt

$ ft propose 001-portrait.prompt.txt prompt.md
./001-portrait.prompt.md
```

Output includes the directory from the input path.

### `ready <path>`

Check whether a file is ready for transformation. Returns `true` if the file
can be processed, `false` otherwise.

This allows the naming tool to signal whether a file requires transformation
based on its type, state, or other criteria specific to the workflow.

```
$ ft ready 001.prompt.txt
true

$ ft ready 001.out.txt
false

$ ft ready unknown.xyz
false
```

Exits `0` on success (regardless of true/false result), non-zero on error.

### `bundles [dir]`

List all browsable bundles in a directory (defaults to current directory).

Each bundle is a prior file (or orphan source) with its associated source files.
Output is tab-separated: `<prior>\t<source1>,<source2>,...`

```
$ ft bundles
153.prompt.md	153.out.txt,153a.out.txt,153b.out.txt
154.prompt.txt	154.out.txt
155.out.txt
```

Bundle rules:
- Each prior file becomes a bundle with its matching sources
- Source files without a matching prior become their own bundle (orphans)
- Sources are comma-separated; field may be empty for orphans
- One bundle per line

This provides everything needed to populate a file browser:
- First column: files to display (priors + orphans)
- Second column: associated sources for each

Exits `0` on success, non-zero on error.

## Types

| Type       | Extension      |
|------------|----------------|
| prompt.txt | `.prompt.txt`  |
| prompt.md  | `.prompt.md`   |
| out.txt    | `.out.txt`     |
| mb         | `.mb`          |
| log        | `.log`         |

## Filename Structure

Filenames follow the pattern: `[index][variation?][-name?][.extension]`

- `index` — Zero-padded digits (e.g., `001`, `0042`)
- `variation` — Lowercase letters for alternates (`a`, `b`, ..., `z`, `za`, `zb`, ...)
- `name` — Optional descriptor, hyphen-prefixed, lowercase alphanumeric with hyphens/underscores
- `extension` — Type-specific extension from table above

## Error Handling

- Non-zero exit code indicates failure
- `stdout` undefined on error
- No guaranteed `stderr` format
