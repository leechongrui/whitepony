# Model & Policy Documentation

## Label Schema
- Multi-label: spam_ad, off_topic, non_visit_rant
- Binary: relevant

## Policy Map
- See policy.yml for mapping model outputs to actions (ALLOW, REVIEW_MANUAL, BLOCK)
- Thresholds per label are configurable in policy.yml
