"""Utility functions for handling timeseries data."""


def make_timeline(df, freq):
    return (
        df.set_index('tweet_time').resample(freq).size().reset_index().rename(
            columns={
                0: 'per_{}_count'.format(freq)
            }).set_index('tweet_time'))
