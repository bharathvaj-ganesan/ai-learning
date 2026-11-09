"""
AUTOENCODER - CARD FRAUD (10 min)

A card issuer does not have a label for every stolen card. It trains on
ordinary swipes only. Each swipe is 4 numbers:

  amount in dollars, hour of the day, miles from home, swipes in the last hour.

Ordinary swipes here happen between 8am and 8pm, within 12 miles of home,
with 1 or 2 swipes an hour. The amount mostly follows how far the trip was.

The encoder squeezes the 4 numbers into 2. The decoder rebuilds the swipe
from only those 2. The loss is "how far is the rebuild from the original?"

An ordinary swipe fits, so the error stays tiny. A stolen-card swipe does
not fit in those 2 numbers, so it comes back looking ordinary and the error
jumps. That jump is the alert. No fraud labels were used.

Run:  python 4_applications/12_autoencoder.py
"""
import torch
import torch.nn as nn

g = torch.Generator().manual_seed(0)
N = 4000
hour = 8 + 12 * torch.rand(N, generator=g)                 # 8am to 8pm
miles = 12 * torch.rand(N, generator=g)                    # within 12 miles
recent = torch.randint(1, 3, (N,), generator=g).float()    # 1 or 2 swipes
amount = 8 + 2.0 * miles + 0.35 * (hour - 8)               # bigger trip, bigger bill
amount = (amount + torch.randn(N, generator=g) * 1.2).clamp(2, 80)
raw = torch.stack([amount, hour, miles, recent], 1)
held, train = raw[:3], raw[3:]

# Never shown during training. The tags are for you, not for the model.
frauds = [
    ([2400, 3, 4, 1], "big charge at 3am"),
    ([22, 14, 850, 1], "ordinary amount, another city"),
    ([420, 2, 6, 1], "big charge at 2am"),
]
fraud_x = torch.tensor([row for row, _ in frauds], dtype=torch.float)

# Dollars and miles are different sizes. Measure each column as
# "how many normal-swipes away from average" so all four count.
mu, sd = train.mean(0), train.std(0)


def norm(x):
    return (x - mu) / sd


LATENT = 2


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, LATENT))
        self.dec = nn.Sequential(nn.Linear(LATENT, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x):
        z = torch.tanh(self.enc(x))          # 2 numbers, kept between -1 and 1
        return z, self.dec(z)


def clock(hour):
    hour = float(hour) % 24
    hh = int(hour)
    mm = int(round((hour - hh) * 60))
    if mm == 60:
        hh = (hh + 1) % 24
        mm = 0
    suffix = "am" if hh < 12 else "pm"
    return f"{hh % 12 or 12}:{mm:02d}{suffix}"


def describe(v):
    n = float(v[3])
    count = f"{n:.1f}".rstrip("0").rstrip(".")
    word = "swipe" if abs(n - 1) < 0.05 else "swipes"
    return (f"${v[0]:.0f} at {clock(v[1])}, "
            f"{v[2]:.1f} miles from home, "
            f"{count} {word} in the last hour")


print("Ordinary swipes are made up here, not loaded from a file.")
print("  hour     8am to 8pm")
print("  miles    0 to 12 from home")
print("  swipes   1 or 2 in the last hour")
print("  amount   about $8 + $2 per mile, a bit more later in the day")
print(f"\n{len(train)} of them are the training set. A few:")
for row in train[:4]:
    print(f"  {describe(row)}")
print(f"\n{len(held)} more are the same kind of swipe, kept out of training.")
print("The fraud rows below are not in this set, and the model is never told they are bad.")

torch.manual_seed(0)
ae = AE()
opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
tn = norm(train)

print("\nTraining. Loss is how far a rebuild is from the swipe it just saw.")
for step in range(2001):
    batch = tn[torch.randint(0, len(tn), (256,))]
    _, hat = ae(batch)
    loss = ((hat - batch) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(f"{step:4d}  {loss.item():.3f}")


def score(x):
    """Rebuild error per swipe, after scaling. High means it did not fit."""
    _, hat = ae(norm(x))
    err = ((hat - norm(x)) ** 2).mean(1)
    back = hat * sd + mu
    return err, back


ae.eval()
with torch.no_grad():
    train_err, _ = score(train)
    cutoff = train_err.max().item() * 10
    print(f"\nWorst ordinary training swipe scored {train_err.max().item():.2f}.")
    print(f"Send a swipe to a person when its error is above {cutoff:.2f}.")

    def show(title, rows, notes):
        err, back = score(rows)
        print(f"\n{title}")
        for i in range(len(rows)):
            call = "review" if err[i] > cutoff else "ok"
            print(f"  {notes[i]}")
            print(f"    swipe    {describe(rows[i])}")
            print(f"    rebuilt  {describe(back[i])}")
            print(f"    error {err[i].item():.2f}  ->  {call}")

    show("Ordinary swipes the model never trained on. These should come back almost unchanged.",
         held, ["held out of training"] * len(held))
    show("Swipes that break the habit. The model was not told these are bad.",
         fraud_x, [note for _, note in frauds])

print("\nThe three bad swipes come back as the same ordinary one:")
print("about $68 around 3pm, about 30 miles out.")
print("The decoder cannot write $2400, 3am, or a trip to another city.")
print("The error is the gap between the swipe and that rebuild.")

"""
TRY THIS
1. Add [28, 13, 5, 1] to the fraud list. It is an ordinary afternoon swipe,
   so it should stay under the cutoff.
2. Change the 2400 / 3am swipe to [28, 13, 4, 1]. The alert should go away.
3. Remove the torch.tanh around the code and rerun. The 2400 row should
   still go to review, but the rebuilt amount, hour, and miles can come out
   impossible. tanh is what keeps the rebuild near an ordinary swipe.

TAKEAWAY: train on ordinary swipes only. The squeeze learns that habit.
Small rebuild error = same habit. Large rebuild error = send it to a person.
No fraud labels were used.
"""
