#yo chai aaja ko aadhar ma voli ko predict garna
#jastai aajai sunny xa vaye voli rain parna sakni chance 20% XA Else SUNNY NAI HUNI 80% XA VANNI TYPE



#states
#yo chai j ni huna sakxa aaile lai warm,cold,high,low vanamna
#sunny rainny


#Transition vanya chai sunny day vaye 80% chance xa sunny nai hos
# ra 20% chance ta rainny hus


#observation vanya chai sunny vaye 80% chance xa ma khushi hunxu kina ni dulna paiyo nara 20%  chance xa ma dulna najani 

import tensorflow as tf
import tensorflow_probability as tfp

#yo chai paxi use garna banako
tfd = tfp.distributions


#yesko meaning chai duita point xa ni aaile huna sakini..sunny huna sakni 80% ra cold ko lagi 20%
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])

#aaha notice garna parni k xa vaney..sab array ma hamle 2ta value deko xam kina ki aaha possible casses 2ta matra xa..either sunny or cold

#yo chai aauta bata rko ma transaition huni probability
#aaile cold xa vani 70%chances xa fere cold nai huni ra 30% chance xa sunny huni
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                        [0.2,0.8]])



# aaha chai 0 ra 15 le indicate garya average temp ho i.e cold ma 0 ra sunny ma 15
#scale ko kam chai temp ko range dini
#average ma +- garera
#i.e cold ko lage 0-5= -5 to +5 ra sunny ko lagi 5 to 25
observation_distribution = tfd.Normal(loc=[0.,15.],scale=[5.,10.])


#num_step le chai aba kati ota extra value predict garna parni indicate garxa
#hamle for a week ko gardaixam so 7 use garya
model = tfd.HiddenMarkovModel(
    initial_distribution= initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps =7
    )


mean = model.mean()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())