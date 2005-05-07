import dstat, glob

### FIXME: Should read /var/log/mail/statistics or /etc/mail/statistics (format ?)

class dstat_sendmail(dstat.dstat):
	def __init__(self):
		self.name = 'sendmail'
		self.format = ('d', 4, 100)
		self.vars = ('queue',)
		self.nick = ('queu',)
		self.init(self.vars, 1)

	def extract(self):
		self.val['queue'] = len(glob.glob('/var/spool/mqueue/qf*'))

# vim:ts=4:sw=4
