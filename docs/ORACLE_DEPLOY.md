# Deploying to Oracle Cloud Free Tier

This guide gets the trader bot running 24/7 on a permanently-free Oracle Cloud VM.

## 1. Create the VM (one time)

1. Sign up at [oracle.com/cloud/free](https://www.oracle.com/cloud/free/). Credit card needed for verification, **not charged** if you stay inside Always Free limits.
2. In the OCI console: **Compute → Instances → Create Instance**.
3. Configure:
   - **Image**: Canonical Ubuntu 22.04 (or 24.04)
   - **Shape**: **Ampere A1.Flex** with 1 OCPU + 6 GB RAM (Always Free). If Ampere is "out of capacity" in your region, fall back to **VM.Standard.E2.1.Micro** (AMD, 1 GB RAM — tighter but workable).
   - **Networking**: leave defaults; assign a public IPv4.
   - **SSH keys**: paste in your public key (`cat ~/.ssh/id_ed25519.pub` on your Mac, or generate one with `ssh-keygen -t ed25519` first).
4. Click Create. Note the **public IP** when it boots.

## 2. Allow outbound HTTPS (usually default)

Oracle's default security list permits all egress, which is all this bot needs. No inbound ports required.

## 3. SSH in and install Docker

```bash
ssh ubuntu@<your-public-ip>

# Install Docker + compose plugin
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker ubuntu
exit                       # log out so the group change takes effect
ssh ubuntu@<your-public-ip>
docker --version           # sanity check
```

## 4. Get the code onto the VM

Pick one:

**A. Via GitHub (recommended once you push it):**
```bash
git clone https://github.com/<you>/trader-bot.git
cd trader-bot
```

**B. Direct copy from your Mac (no GitHub needed):**
```bash
# Run this on your Mac:
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='.env' \
    ~/Documents/trader-bot/ ubuntu@<your-public-ip>:~/trader-bot/
# then SSH back in:
ssh ubuntu@<your-public-ip>
cd trader-bot
```

## 5. Configure secrets

```bash
cp .env.example .env
nano .env                  # paste your Alpaca paper keys
```

## 6. Build and run

```bash
docker compose build       # ~3-5 min the first time
docker compose run --rm trader python scripts/train_models.py   # train the ML model once
docker compose up -d       # start the bot in the background
docker compose logs -f     # tail logs (Ctrl-C to detach; container keeps running)
```

`restart: unless-stopped` in `docker-compose.yml` means the bot survives VM reboots automatically — no systemd unit needed.

## 7. Day-to-day commands

```bash
docker compose logs -f --tail=200       # live logs
docker compose ps                       # is it running?
docker compose restart                  # after editing config.yaml
docker compose down                     # stop the bot
docker compose pull && docker compose up -d --build   # update + restart
```

## 8. Hardening (do these before flipping to live mode)

- **Limit SSH** in Oracle's security list to your home IP only.
- **Rotate keys** if you ever paste `.env` content anywhere unsafe.
- **Enable automatic OS patches**: `sudo apt install unattended-upgrades`.
- **Backup `models/` and `data_cache/`** to S3 or your laptop periodically — they're worth keeping.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker compose build` killed mid-pip | Low RAM. Add swap: `sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |
| Container restarts every cycle | Check `docker compose logs` — usually a missing `.env` value or a yfinance rate-limit |
| `permission denied` on volumes | The image runs as UID 1000; on Oracle's Ubuntu image `ubuntu` is also UID 1000, so this should "just work". If not: `sudo chown -R 1000:1000 models data_cache logs` |
