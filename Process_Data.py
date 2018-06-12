
# coding: utf-8

# In[29]:


import numpy as np
import mido
from sklearn.preprocessing import MinMaxScaler


# In[2]:


def Read_File(path):
    pattern = mido.MidiFile(path)
    return pattern


# In[7]:


def Get_All_Msgs(pat):
    l = []
    found = False
    for i, track in enumerate(pat.tracks):
        for msg in track:
            if not msg.is_meta and msg.type == 'note_on':
                l.append(msg)
                found = True
        if found:
            break

    return l


# In[46]:


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
    t = np.array([t], dtype=np.float32)
    
    return note, velocity, t


# In[23]:


def Get_Categorical(v, num_classes):
    cat_mat = np.zeros( (v.shape[0], num_classes) )
    
    for i in range(v.shape[0]):
        cat_mat[i][ v[i] ] = 1
    
    return cat_mat


# In[17]:


def Write_File(note, vel, t):
    
    new_midi = mido.MidiFile()
    track = mido.MidiTrack()
    new_midi.tracks.append(track)

    track.append(mido.Message('program_change', program=0, time=0))

    for i in range( note.shape[0] ):
        track.append( mido.Message( 'note_on' ,note = note[i], velocity = vel[i], time = t[i]) )
    
    return new_midi


# In[65]:


def Processed_Data(path, scaler = None):
    
    pat = Read_File(path)
    l = Get_All_Msgs(pat)
    note, velocity, t = Get_Data(l)
    note = Get_Categorical(note, 128)
    velocity = Get_Categorical( velocity, 128 )
    
    data = {
        'note': note,
        'velocity': velocity
    }
    
    t = np.reshape(t, (-1,1) )
    
    if not scaler:
        time_scaler = MinMaxScaler( feature_range=(0,1) )
        t_scaled = time_scaler.fit_transform( t )
        
        data['scaler'] = time_scaler
        data['MulFactor'] = time_scaler.scale_[0]
        data['AddFactor'] = time_scaler.min_[0]
        
    else:
        t_scaled = scaler.fit( t )
        data['scaler'] = scaler
        data['MulFactor'] = scaler.scale_[0]
        data['AddFactor'] = scaler.min_[0]
    
    data['time'] = t_scaled
    
    ## Time will be obtained by multiplying it by MulFactor and adding AddFactor
        
    return data

