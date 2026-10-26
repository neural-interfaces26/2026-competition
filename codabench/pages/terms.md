# Participation rules

## Rule 1: Anyone can enter. Organisers cannot win prizes

Industry, academia, students, independents - all eligible, and cross-institution teams are encouraged. Organisers and their direct team members may submit, but their entries appear as Organizers on the leaderboard and are ineligible for cash prizes.

## Rule 2: One team may enter all four tracks

Each participant submits through their own Codabench account and has an individual submission quota for each track. For team entries, submissions are attributed to the participant's Codabench organization. Solo participants count as one-person teams. See [automatic leaderboard selection](#which-submission-appears-on-the-leaderboard-and-counts-for-the-final-ranking).

## Rule 3: How many submissions can I make?

Both phases have separate quotas **per participant and per track**:

- **Warm-up:** 5 submissions per participant per day per track, with **100 submissions total per participant per track**.
- **Sealed:** 1 submission per participant per day per track, with **5 submissions total per participant per track**.

The five-submission sealed cap limits repeated tuning against hidden test data. Sealed evaluation measures generalization, not blind trial-and-error optimization on the test set.

Warm-up closes Sunday, November 1 at 23:59 CET (UTC+01:00). Sealed runs Thursday, November 5 at 00:01 through Saturday, November 21 at 23:59 CET (UTC+01:00, Europe/Paris). No new submissions are accepted during the transition. See the [Timeline](timeline.md) for the full schedule.

### What counts as a submission in my quota?

Each ZIP accepted for evaluation is a submission, even if you upload the same model again. Both the daily and total limits follow these rules:

- **Queued or running:** reserves a quota slot. A submission with final status `Failed` does not count.
- **Finished or cancelled by you:** counts as one submission.
- **Deleted:** deleting a counted submission does not restore its quota slot.
- **Organizer interventions:** cancellations and reruns required to fix platform incidents do not reduce your allowance. Organizers handle any necessary quota correction.

### Which submission appears on the leaderboard and counts for the final ranking?

For each phase and track, Codabench automatically displays the best eligible score from your team, including all members' submissions. Solo participants count as one-person teams. A better score replaces the displayed entry, with no manual publication needed. The submission currently displayed on the leaderboard cannot be deleted.

Only the **best eligible sealed score** from you or your team is retained for the final ranking, subject to the reproducibility audit. Warm-up scores are indicative and never count toward that ranking.

## Rule 4: Pre-train on any public dataset. Declare all of it

Training data is suggested to stay within the track. Public datasets may be used when their original terms permit competition use. This includes NSRR datasets under an active licence, as clarified in the [Track 03 Scientific FAQ](scientific_faq_sleep.md#may-nsrr-or-other-public-sleep-datasets-be-used). The sealed test split and non-permitted closed clinical datasets are not allowed. Original dataset and model licences remain in force. Declare every external corpus, and a compute estimate, in the method description that ships with your final submission -- the audit cross-checks it against a rerun of your submitted training pipeline and declared configuration.

## Rule 5: Training compute is uncapped. Evaluation is limited to 60 minutes on one GPU

Train on whatever you have. The evaluation container must complete its test pass within 60 minutes on one A100 GPU. Evaluations that exceed this execution limit time out.

Codabench limits each profile to 15 GB of submission storage shared across all tracks and uploads. This makes 15 GB the theoretical maximum for one submission, but the practical limit is the space remaining in the profile. Participants may delete unused archived submissions under **Profile → Resources**, but a submission displayed on a leaderboard cannot be deleted.

## Rule 6: Cheating attempts

Any attempt to cheat, including but not limited to using non-permitted clinical data, accessing the sealed test split, or modifying the scoring harness, will result in immediate disqualification from the competition. This decision will be made by the organisation committee and will be final.

## Rule 7: The top three per track must replay within ±2σ

After the sealed final closes on Nov 21, 2026 at 23:59 CET (UTC+01:00), the top three teams in every track go through a reproducibility audit during Nov 22–30: we rerun your training pipeline from the committed configuration and rescore the resulting weights against the sealed split. Scores within ±2σ of your submitted number stay on the prize roster; scores outside the tolerance drop off the prize roster but remain on the public board for context. The organizing committee will chair the audit.

## Rule 8: Reproducible top teams will be announced during the NeurIPS workshop

Top-ranked teams that pass the audit and submit a method description, training and inference code, and pre-training disclosures will be announced during the NeurIPS workshop on Dec 11–12. Winner notifications are targeted for Dec 1, after audit validation. This opportunity is offered to teams with complete, reproducible artifacts.

## Submission ownership and audit use

Participants retain the intellectual property rights they hold in their submitted code, models, and related materials. Finalists must provide the materials required for the reproducibility audit and grant the auditing committee permission to access and run them solely to verify the submission and detect cheating. This limited audit permission does not allow the organizers to publish, share, modify, commercialize, or otherwise use those materials outside the audit. The organizers encourage participants to release their solutions through established open-source libraries after the competition, but that decision remains entirely with each participant.

## Sealed-phase feedback

Submission status and aggregate leaderboard scores remain visible. Execution logs, prediction files and scoring-step output are hidden during the sealed phase. Validate your ZIP locally and in warm-up beforehand; sealed logs are not available for debugging. See the [Submission Guide](participate.md) for local validation.

## Reference baseline performance

Reference EEGNet and REVE scores for warm-up and sealed evaluation will be added in due time to the [baseline performance sections of the participant guide](https://neural-interfaces26.github.io/participant-guide.html#baselines), alongside the corresponding training procedures.
