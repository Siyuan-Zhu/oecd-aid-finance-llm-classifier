# Data and attribution

## Source

The sample comes from the OECD Creditor Reporting System (CRS), the DAC's project-level database of official
development assistance. This repo uses the 2024 flows file and the official list of purpose codes.

- Project data: `CRS 2024 data.zip`, <https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS%202024%20data.zip>. 46,495,852 bytes,
  SHA-256 `9e1387b4f3981854f9de786758e482add7a802e207f94d1bddaad9cef51fb9b5`, dataflow version v20261006, read on
  23 Oct 2026. Inside is `CRS 2024 data.txt`: UTF-8, pipe-delimited, every field double-quoted, CRLF, 102 columns.
- Purpose-code list: `DAC-CRS-CODES.xlsx`, sheet "Purpose codes",
  <https://webfs.oecd.org/oda/DataCollection/Resources/DAC-CRS-CODES.xlsx>. SHA-256
  `eb5d320eeebb93f97f6b4a8e0ff4957cf923a4a59efa35f7bb3fab7d7c5f7b52`, read on 23 Oct 2026. The sheet's header says "Last
  updated in March 2025, valid for reporting on 2025 flows". `config/purpose_codes.json` holds the whole list (49 sector
  headings, 296 five-digit codes with names, descriptions, parent sector and a voluntary-code flag), rebuilt by
  `scripts/build_taxonomy.py`. Code names are copied from the workbook, not typed by hand.
- Suggested citation, in the form the OECD terms ask for: OECD (2026), Creditor Reporting System (CRS) 2024 data,
  <https://webfs-dcd.oecd.org/files/dotStat/DSD_CRS/CRS%202024%20data.zip> (accessed on 23 October 2026).

## Terms of use

The OECD terms and conditions, section 3 "Data" (<https://www.oecd.org/en/about/terms-conditions.html>), say:

> you can extract from, download, copy, adapt, print, distribute, share and embed Data for any purpose, even for
> commercial use. You must give appropriate credit to the OECD by using the citation associated with the relevant Data,
> or, if no specific citation is available, you must cite the source information using the following format: OECD (year),
> (dataset name),(data source) DOI or URL (accessed on (date)).

The same section adds: "Data may be subject to restrictions beyond the scope of these Terms and Conditions, either because
specific terms apply to those Data or because third parties may have ownership interests." The project titles and
descriptions are free text written by donor agencies, so third-party ownership is possible. This repo commits a sample of
3,227 of them with the credit above. That is my reading of the terms, not legal advice. If you reuse the
sample, give the same credit and check the terms yourself. Anyone who finds a description that should not be here can
open an issue and it will be removed.

The OECD did not produce, check or endorse this repo, the selection of rows, or any result computed from them.

## What is committed

`data/crs_health_2024_sample.csv`: 3,227 rows, about 2.3 MB, SHA-256
`31a8a33651eeaf5b8be36b1be93530b6e55e7452d8bf3dec72b3f25f3c5eaf57` (pinned in `src/data.py`; a mismatch stops the run).
Columns: `crs_id`, `year` (all 2024), `donor_name`, `sector_code`, `purpose_code`, `title`, `long_description`. These are
the OECD's `CrsID`, `Year`, `DonorName`, `SectorCode`, `PurposeCode`, `ProjectTitle` and `LongDescription` fields with
whitespace collapsed. Text is otherwise left as published, including the mojibake (for example "CooperaciÃ³n") and the `?`
characters that stand in for some non-ASCII letters in the source. `CrsID` is not unique per text (the same project can
appear with several descriptions), so `src/data.py` numbers repeats (`id` = `crs_id`-1, `crs_id`-2, ...).

The model input is the title and the long description joined as `title + ". " + description`. The 3-digit sector is an
input. The target is the 5-digit purpose code.

## Filters and row counts

`scripts/build_sample.py` prints these counts when it runs; they are copied here. Scope is the three health sectors,
121 Health, General, 122 Basic Health and 123 Non-communicable diseases (NCDs).

| Step | Rows left | Removed |
| :--- | ---: | ---: |
| Data lines in the bulk file | 370,264 | |
| Skipped as malformed (wrong field count, embedded quote or delimiter) | 370,242 | 22 |
| Sector code 121, 122 or 123 | 38,860 | 331,382 |
| Purpose code is a 5-digit child of the sector | 38,854 | 6 |
| Long description has at least 20 words (letters regex) | 20,245 | 18,609 |
| English by stopword heuristic | 17,029 | 3,216 |
| Boilerplate title or description ("TC AGGREGATED ACTIVITIES") | 17,029 | 0 |
| Text carries more than one purpose code: all its rows dropped | 8,119 | 8,910 |
| One row per distinct text | 6,307 | 1,812 |
| Codes with at least 30 distinct texts | 6,271 | 36 |
| Capped at 250 per code (seed 42) | 3,227 | 3,044 |

Notes on the steps:

- **Stray labels.** Six rows carry the 3-digit sector as their purpose code ("121", "122", "123"). They are not 5-digit
  codes and are removed.
- **Language.** An English stopword count must be at least as high as the French and Spanish counts and at least 10% of
  the words. It is a heuristic and will pass some English-heavy mixed text and fail some terse English.
- **Boilerplate.** The pattern is an exact match on "TC AGGREGATED ACTIVITIES" (1,278 health rows). The 20-word filter had
  already removed all of them, so this step removes nothing; it stays as a guard.
- **Duplicates.** 72.7% of health descriptions in the bulk file are repeats of another row's text (the same programme
  reported across donors, recipients, agencies or years). One row is kept per distinct title-plus-description text.
- **Conflicts.** If the same text appears under more than one purpose code, every row with that text is dropped, 8,910
  rows in all. This removes the ambiguous cases, so the sample is cleaner than the raw file and the task here is easier
  than coding every project the OECD receives.
- **Cap.** Up to 250 texts per code are kept, drawn with a fixed seed (`random.Random("42:<code>")`).

## Codes in the sample, and codes dropped

| Sector | Code | Name | Distinct texts available | In sample | Train | Val | Test |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 121 | 12110 | Health policy and administrative management | 1,089 | 250 | 150 | 50 | 50 |
| 121 | 12181 | Medical education/training | 129 | 129 | 78 | 25 | 26 |
| 121 | 12182 | Medical research | 580 | 250 | 150 | 50 | 50 |
| 121 | 12191 | Medical services | 512 | 250 | 150 | 50 | 50 |
| 122 | 12220 | Basic health care | 823 | 250 | 150 | 50 | 50 |
| 122 | 12230 | Basic health infrastructure | 245 | 245 | 147 | 49 | 49 |
| 122 | 12240 | Basic nutrition | 523 | 250 | 150 | 50 | 50 |
| 122 | 12250 | Infectious disease control | 679 | 250 | 150 | 50 | 50 |
| 122 | 12261 | Health education | 529 | 250 | 150 | 50 | 50 |
| 122 | 12262 | Malaria control | 295 | 250 | 150 | 50 | 50 |
| 122 | 12263 | Tuberculosis control | 264 | 250 | 150 | 50 | 50 |
| 122 | 12264 | COVID-19 control | 199 | 199 | 120 | 40 | 39 |
| 122 | 12281 | Health personnel development | 80 | 80 | 48 | 16 | 16 |
| 123 | 12310 | NCDs control, general | 36 | 36 | 22 | 7 | 7 |
| 123 | 12340 | Promotion of mental health and well-being | 163 | 163 | 98 | 32 | 33 |
| 123 | 12350 | Other prevention and treatment of NCDs | 45 | 45 | 27 | 9 | 9 |
| 123 | 12382 | Research for prevention and control of NCDs | 80 | 80 | 48 | 16 | 16 |
| | | Total | 6,271 | 3,227 | 1,938 | 644 | 645 |

Sectors 121 to 123 hold 20 five-digit codes in the OECD list. The sample has 17. The other three:

- **12196 Health statistics and data** (sector 121) is a voluntary code, not a core CRS code. It has no rows in the 2024
  file.
- **12320 Tobacco use control** (sector 123) has 10 distinct texts after the filters, below the minimum of 30. Dropped.
- **12330 Control of harmful use of alcohol and drugs** (sector 123) has 26 distinct texts, below the minimum. Dropped.

The smallest kept codes are 12310 (36 texts, 7 in test), 12350 (45, 9 in test) and 12281 (80, 16 in test). Scores on
those codes rest on a handful of examples and move a lot with one more or one fewer right answer.

## Train, validation and test split

60/20/20 into train (1,938), validation (644) and test (645), stratified by purpose code with scikit-learn's
`StratifiedGroupKFold` (5 folds, shuffled, seed 42; one fold is test, one is validation, three are train).

The groups are near-duplicate clusters. The model input text (title plus description) of every sample row is turned into
a TF-IDF vector with the vectoriser in `src/settings.py` (unigrams and bigrams, English stop words removed, sublinear term
frequency), fitted on the whole sample. Two rows are joined when their cosine similarity is at least
`NEAR_DUPLICATE_COSINE = 0.8` (`src/data.py`). The groups are the connected components of that relation (union-find), so a
chain of similar texts forms one group, and a whole group goes to one split. Splitting is then done on groups.

What this guarantees: for this vectoriser, every val or test text has cosine below 0.8 to every train text. The highest
value is 0.798 from a val text and 0.789 from a test text to its nearest train text. `tests/test_data.py` checks this
through `load_splits()`. It does not mean no paraphrase of a test text is in train: pairs just under 0.8 are still
allowed to sit in different splits, and a different vectoriser could score a pair differently.

Group statistics over the 3,227 rows: 2,928 groups, of which 2,857 are single texts. The largest group has 28 texts, all
under 12261 Health education. No group is large enough to swallow a code, so no special
handling was needed; every code keeps its 60/20/20 shares to within a text or two. The three sectors split 528/176/176
(121), 1,215/405/404 (122) and 195/64/65 (123), train/val/test.

## Rebuild

```bash
pip install openpyxl                      # script-only dependency
python scripts/build_taxonomy.py --download
python scripts/build_sample.py --download # fetches the 46 MB zip into data/raw/ (gitignored) and checks its SHA-256
```

The OECD host sits behind Cloudflare, which blocks `curl`. `build_sample.py` uses Python `urllib` with a plain
User-Agent and an `Upgrade-Insecure-Requests: 1` header, which a HEAD request on 23 Oct 2026 accepted (HTTP 200,
46,495,852 bytes). A fresh build from the same zip reproduces the committed CSV byte for byte. The OECD may publish a
newer version of the file later, in which case the SHA-256 check fails on purpose.
