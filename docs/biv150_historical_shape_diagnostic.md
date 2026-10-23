# Historical execution-shape diagnostic

Historical base arm based on 9a43eb1, deliberately reproducing the old W8A8 comparison.
This is not team integration work and must not be merged into main. Team main
remains3283672. Changes only add the opt-in host execution-shape recorder;
no operator selection, model weights, scheduling budget or graph mode changes.
The full record is not a performance score. Loader env activates the recorder
only in a separate diagnostic process, never in the ongoing formal gate-only run.
