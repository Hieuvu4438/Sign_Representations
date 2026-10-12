#!/usr/bin/env bash
# Cron watchdog for the PhonoJEPA runner (installed by crontab, every 10 min):
# restarts scripts/schedule.py when it is not running (crash / server reboot) and the plan is not finished.
# The runner itself waits for a free GPU, skips finished jobs (runtime/schedule_state.json) and resumes training.
# Disable: touch runtime/STOP   (or remove the crontab line).
cd /home/haipd/Sign_Representations/method/phonojepa || exit 0
[ -f runtime/STOP ] && exit 0
grep -q "ALL DONE" runtime/logs/schedule.log 2>/dev/null && exit 0
flock -n runtime/schedule.lock -c true 2>/dev/null || exit 0          # lock held -> runner alive
echo "$(date -Iseconds) watchdog: runner not running -> restart" >> runtime/logs/watchdog.log
setsid nohup /home/haipd/Sign_Representations/.venv-signrep/bin/python scripts/schedule.py >> runtime/logs/schedule.log 2>&1 < /dev/null &
