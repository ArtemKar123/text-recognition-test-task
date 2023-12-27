# Solution
## Model
EasyOCR is used for detection and recognition. Of course, we can do better by fine-tuning it for our IDE oriented task, but for now the default model seems to work just fine.
## Postprocessing
Results of recognition are postprocessed with an assumption, that the data consists of IDE screenshots (or somewhat similar).
Basically, postprocessing consists of combining text boxes and adding new lines and tabs to them.
## Side-note
There were some typos in test data, so there is a commit that fixes them.
# [Happy New Year!](https://www.youtube.com/watch?v=hJresi7z_YM)
