# Participation rules

## Rule 1: Anyone can enter. Organisers cannot win prizes

Industry, academia, students, independents - all eligible, and cross-institution teams are encouraged. Organisers and their direct team members may submit, but their entries appear as Organizers on the leaderboard and are ineligible for cash prizes.

## Rule 2: One team may enter all four tracks

Each participant submits through their own Codabench account and has an individual submission quota. For team entries, those submissions are attributed to the participant's Codabench organization and ranked as team submissions.

## Rule 3: Five submissions a day in warm-up, one per day in the sealed final

Warm-up (Sep 21 - Oct 25, 2026): 5 submissions per participant per day. Sealed final (Oct 28 - Nov 21, 2026, AoE): one submission per participant per day, so the prize roster reflects deliberate iteration rather than lottery search. The final NeurIPS ranking uses sealed submissions only.

## Rule 4: Pre-train on any public dataset. Declare all of it

Training data is suggested to stay within the track, but pre-training on any publicly available, redistributable dataset is allowed. The sealed test split is never allowed. Closed clinical datasets are not allowed. Original dataset and model licenses remain in force. Declare every external corpus, and a compute estimate, in the method description that ships with your final submission -- the audit cross-checks it against a rerun of your submitted training pipeline and declared configuration.

## Rule 5: Training compute is uncapped. Inference is 60 minutes on one GPU

Train on whatever you have. The single constraint is inference: the scoring container must complete a full test pass in under 60 minutes on one A100 instance, so that audit cost stays bounded and per-team runtime stays comparable.

Codabench limits each profile to 15 GB of submission storage shared across all tracks and uploads. This makes 15 GB the theoretical maximum for one submission, but the practical limit is the space remaining in the profile. Participants may delete unused archived submissions under **Profile → Resources**, but a submission displayed on a leaderboard cannot be deleted.

## Rule 6: Cheating attempts

Any attempt to cheat, including but not limited to using closed clinical data, accessing the sealed test split, or modifying the scoring harness, will result in immediate disqualification from the competition. This decision will be made by the organisation committee and will be final.

## Rule 7: The top three per track must replay within ±2σ

After the sealed final closes on Nov 21, 2026, the top three teams in every track go through a reproducibility audit: we rerun your training pipeline from the committed configuration and rescore the resulting weights against the sealed split. Scores within ±2σ of your submitted number stay on the prize roster; scores outside the tolerance drop off the prize roster but remain on the public board for context. The organizing committee will chair the audit.

## Rule 8: Reproducible top teams will be announced during the NeurIPS workshop

Top-ranked teams that pass the audit and submit a method description, training and inference code, and pre-training disclosures will be announced during the NeurIPS workshop. This opportunity is offered to teams with complete, reproducible artifacts.

## Submission ownership and audit use

Participants retain the intellectual property rights they hold in their submitted code, models, and related materials. Finalists must provide the materials required for the reproducibility audit and grant the auditing committee permission to access and run them solely to verify the submission and detect cheating. This limited audit permission does not allow the organizers to publish, share, modify, commercialize, or otherwise use those materials outside the audit. The organizers encourage participants to release their solutions through established open-source libraries after the competition, but that decision remains entirely with each participant.
