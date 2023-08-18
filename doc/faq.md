# Frequently Asked Questions

__Something happened and FlightGoggles never closed. What should I do?__

Our Python software operates by launching FlightGoggles in a subprocess.
Nominally, Python will track this subprocess and shutdown FlightGoggles prior to shutting itself down.
However, in some situations FlightGoggles may remain open after closing Python (e.g., an error in Python).

In these situations, you can safely close FlightGoggles by clicking on the `x` on the window.
Alternatively you can shut it down using linux tools such as `kill` or `pkill`, which may be necessary if you are operating on a server or remote machine.

__I'm getting an error about a port (or address) already being in use. What should I do?__

If you observe this error, the most likely cause is that you have a python or FlightGoggles already running that is connected to that port.
This may have happened due to an error or some issue when previously running Python with FlightGoggles.
In this case, you will have to identify the process ID (`pid`) of your previous Python (`ps aux | grep python`) or FlightGoggles (`ps aux | grep FlightGoggles`) instance.
Then, use `kill` to stop the process.

Alternatively, it may be possible you are trying to use a port reserved by another system.
In this case, select a new port to use.

__The depth camera in FlightGoggles looks weird. Is it broken?__

Short answer: probably not! 

Below is an example screen shot showing an RGB camera image on top and a depth image below.
Note the stripped effect on the depth image, which is perfectly normal.
Unity is embedding depth information into the three RGB channels, so you see the result of that embedding.

![FlightGoggles Image](images/flight-goggles-depth-example.png)
