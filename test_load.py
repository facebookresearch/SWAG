print('Testing RegNetY128gf')
from models import regnety_128gf
model = regnety_128gf()


print('Testing RegNetY128gf IN1k')
from models import regnety_128gf_in1k
model = regnety_128gf_in1k()


print('Testing RegNetY32gf IN1k')
from models import regnety_32gf_in1k
model = regnety_32gf_in1k()


print('Testing RegNetY16gf IN1k')
from models import regnety_16gf_in1k
model = regnety_16gf_in1k()


print('Testing ViT H/14 IN1k')
from models import vit_h14_in1k
model = vit_h14_in1k()


print('Testing ViT B/16 IN1k')
from models import vit_b16_in1k
model = vit_b16_in1k()


print('Testing ViT L/16 IN1k')
from models import vit_l16_in1k
model = vit_l16_in1k()