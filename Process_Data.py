
# coding: utf-8

# In[23]:


import numpy as np
import mido
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


# In[2]:

time = []

def Read_File(path):
    pattern = mido.MidiFile(path)
    return pattern


# In[3]:


def Get_All_Msgs(pat):
    l = []
    found = False
    for i, track in enumerate(pat.tracks):
        for msg in track:
            if not msg.is_meta and msg.type == 'note_on':
                l.append( msg )
                found = True
        if found:
            break
            
    return l


# In[4]:


def Get_Data(Msgs):
    note = []
    velocity = []
    t = []
    
    for msg in Msgs:
        note.append( msg.note )
        velocity.append( msg.velocity )
        t.append ( msg.time )
    
    note = np.array(note, dtype=np.int32)
    velocity = np.array(velocity, np.int32)
    t = np.array([t], dtype=np.float32 )
    
    return note, velocity, t


# In[5]:


def Get_Categorical(v, num_classes):
    cat_mat = np.zeros( (v.shape[0], num_classes) )
    
    for i in range(v.shape[0]):
        cat_mat[i][ v[i] ] = 1
    
    return cat_mat


# In[6]:


def Write_File(note, vel, t):
    
    new_midi = mido.MidiFile()
    track = mido.MidiTrack()
    new_midi.tracks.append(track)

    track.append(mido.Message('program_change', program=0, time=0))

    for i in range( note.shape[0] ):
        track.append( mido.Message( 'note_on' ,note = note[i], velocity = vel[i], time = t[i]) )
    
    return new_midi


# In[13]:

def Processed_Data(path):
    
    pat = Read_File(path)
    l = Get_All_Msgs(pat)
    note, velocity, t = Get_Data(l)
    note = Get_Categorical(note, 128)
    velocity = Get_Categorical( velocity, 128 )
    
    t = np.reshape(t, (-1,1) )
    
    for i in range(t.shape[0]):
        time.append(t[i][0])
    
    split = int(0.8 * note.shape[0])
    
    data = {
        'note_train': note[:split],
        'note_test': note[split:],
        'vel_train': velocity[:split],
        'vel_test': velocity[split:],
        'time_train': t[:split],
        'time_test': t[split:]
    }
    
    ## Time will be obtained by multiplying it by MulFactor and adding AddFactor
        
    return data


# In[24]:


data_dir = './mozart/'
song_files = os.listdir(data_dir)


# In[25]:


for song in song_files:
    if song[-3:] != 'mid':
        continue
    data = Processed_Data(data_dir+song)
    with open('./Training_Data/'+ song[:-3] + 'pkl', 'wb') as f:
        pickle.dump(data, f)
    print ( song[:-4] , " Done", len(time) )

print ("Preparing Training Data")
    
time = np.array(time)
time = np.reshape( time, (-1,1) )
time_scaler = MinMaxScaler( feature_range=(0,1) )
t_scaled = time_scaler.fit_transform( time )

train_data = {
    'scaler': time_scaler,
    'MulFactor': time_scaler.scale_[0],
    'AddFactor': time_scaler.min_[0]
}

with open('./Training_Data/train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)

