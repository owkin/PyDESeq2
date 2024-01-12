import time


def timeit(func):
    """
    A decorator function that measures the execution time of the decorated function.

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The decorated function.

    Examples
    --------
    >>> @timeit
    ... def slow_function(n):
    ...     # Simulate a slow function by sleeping for n seconds
    ...     time.sleep(n)
    ...     return n
    ...
    >>> slow_function(2)
    Function took 2.000062942504883 seconds to execute
    2
    """

    def wrapper(*args, **kwargs):
        """
        Internal wrapper function that measures the execution time of the decorated function.

        Parameters
        ----------
        *args : tuple
            Positional arguments to be passed to the decorated function.

        **kwargs : dict
            Keyword arguments to be passed to the decorated function.

        Returns
        -------
        any
            The result returned by the decorated function.

        Notes
        -----
        The wrapper function calculates the time taken to execute the decorated function
        by capturing the start and end times using the `time.time()` function.
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func} took {end_time - start_time} seconds to execute.")
        return result

    return wrapper
