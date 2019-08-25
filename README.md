# neural_sims
A package for generating and analyzing neural data made in conjunction with
[The Katz Lab](https://sites.google.com/a/brandeis.edu/katzlab/). The main goal of this
work is to be able to generate neural firing data that exhibits a 'state change', or a
change in one or many neurons' firing rates. When processing a taste on the tongue, the brain
exhibits multiple state changes signifying different processing steps (i.e. "something touched
my tongue" or "do I like this flavor?").  Each step can be separated by observing the firing rates
of a few neurons in the brain.  When a state change occurs, many neurons in the brain will change their
firing rates even slightly as other neurons feed them more or less action potentials.  We can estimate the
time of a state change by observing that many neurons have changed their firing rates within a few milliseconds
of each other.

By generating timesteps of fake neural data that exhibits synced firing rate changes, this work can be used to
train models that predict the time of a state change in real neural data. This is currently determined
via eye-balling the data to see where many of the firing histograms change in unison.
See [PyHMM](https://github.com/abuzarmahmood/PyHMM) for a few models that benefit from this.
