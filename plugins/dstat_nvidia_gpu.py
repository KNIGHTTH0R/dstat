### Author: Vasilis Vryniotis <bbriniotis@datumbox.com>

class dstat_plugin(dstat):
    """
    Total GPU usage for NVIDIA cards. Requires the nvidia-ml-py package.

    Usage: dstat --nvidia-gpu
    """

    def __init__(self):
        self.name = 'gpu nvidia'
        self.nick = ('gpu',)
        self.vars = ('gpu',)
        self.type = 'p'
        self.width = 5
        self.scale = 1

    def check(self):
        import pynvml
        pynvml.nvmlInit()

    def extract(self):
        self.val['gpu'] = self._getTotalUsage(10)

    def _getUsagePerGPU(self, samples):
        import pynvml
        usage = {}
        deviceCount = pynvml.nvmlDeviceGetCount()
        for iter in range(0, samples):
            for i in range(0, deviceCount):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                if i not in usage:
                    usage[i] = 0.0
                usage[i] += pynvml.nvmlDeviceGetUtilizationRates(handle).gpu / float(samples)
        return usage

    def _getTotalUsage(self, samples):
        usage = self._getUsagePerGPU(samples)
        return sum(usage.values()) / len(usage)


