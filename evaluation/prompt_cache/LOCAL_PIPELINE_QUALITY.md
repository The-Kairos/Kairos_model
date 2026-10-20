# Local source-based quality spot review

Reviewed the first-trial pair for each input with layout labels hidden. The
bundle contained the summary, highlights, nine required answers, three selected
scene reports, source speech, and selected visual evidence. Additional source
scenes were inspected to check disputed claims. Findings were recorded in
`local_review/findings_label_hidden.json` before reading `local_review/mapping.json`.
No external quality judge or additional data export was used.

This covers three of nine pairs, not every claim in all eighteen outputs.

| Input | Finding after decoding labels |
| --- | --- |
| Speech | Both capture the main topic. The original layout incorrectly says the clip ends with cross-border unity; the source continues to personal identity, education, women's rights, and peace. The cache-friendly answer captures that ending more accurately. Both have a sibling-rivalry highlight whose time range does not cover the corresponding source passage. |
| Story | Both preserve the first-day-of-school topic. Both introduce a sister identity from noisy context without adequate explicit support. The cache-friendly version attributes bow-tie criticism to Missy despite nearby speech saying "Please trust Mommy." The original and cache-friendly versions each have a highlight with poor event-to-time alignment. |
| Visual | Both retain the main dog/car/residential content and the stationary-car ending. Unknown quote answers are justified by the absence of speech. The cache-friendly version includes secondary cat/roof details present in source captions; the original is more selective. No decisive factual difference in this spot sample. |

Both layouts also make interpretive scene-context statements that are not always
clearly marked as inference. No consistent quality winner is established.

Automated checks across all eighteen runs show that every scene and its order
were retained; each report contains 37 questions and substantive highlights,
timeline events, and clips. Speech and story runs have no fully missing required
answers. Visual runs have three to five unknown required answers, which must be
assessed against the sparse evidence rather than treated as automatic errors.

All numeric final timestamp fields match time values in the saved evidence.
This does **not** prove that the right event was assigned to that time: the spot
review found alignment errors despite passing the membership check.

The test does not establish equal factual quality. A faster result should not
be accepted merely because its field counts and timestamp syntax pass.

Published summary only. Raw outputs, review bundles, and request traces referenced above remain in the local ignored run directory.
