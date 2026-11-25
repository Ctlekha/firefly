import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title='Firefly Garden',page_icon='✨',layout='wide')
st.markdown('''<style>
.stApp {background: radial-gradient(ellipse at top,#24463c,#101f25 65%); color:#f4f6df;}
[data-testid="stSidebar"] {background:#192f32;}
h1,h2,h3 {color:#e9f5ae!important;} p,label {color:#e1eadb!important;}
[data-testid="stMetric"] {background:#263e3b;border:1px solid #47604b;border-radius:18px;padding:15px;}
.stButton button {border-radius:24px;background:#e5efad;color:#20372d;border:0;font-weight:700;}
.stButton button p {color:#20372d!important;}
</style>''',unsafe_allow_html=True)
st.title('✨ Firefly Garden')
st.caption('A tiny glowing world of swarm intelligence • Interactive learning lab')
st.markdown('Watch little fireflies search for a better solution, one glow at a time. 🌿')
with st.sidebar:
    st.header('🌱 Garden settings')
    scenario=st.selectbox('Learning scenario',['Moonlit garden','Strawberry stall location'])
    n=st.slider('Number of fireflies',5,40,15)
    alpha=st.slider('Randomness (α)',0.,1.,0.18,0.01)
    beta=st.slider('Base attractiveness (β₀)',0.,1.,0.8,0.05)
    gamma=st.slider('Light absorption (γ)',0.,5.,1.,0.1)
    speed=st.slider('Animation delay (seconds)',0.1,1.,0.3,0.1)
    limit=st.slider('Maximum iterations',10,200,60)
    st.caption('Try changing γ to see how distance affects attraction.')

def fitness(p):
    p=np.asarray(p)
    if scenario=='Moonlit garden':
        return np.exp(-np.sum((p-[7.5,7.0])**2,axis=-1)/8)
    return np.maximum(np.exp(-np.sum((p-[7.5,7.0])**2,axis=-1)/5),0.72*np.exp(-np.sum((p-[2.5,3.0])**2,axis=-1)/4))

def reset():
    p=np.random.default_rng().uniform(0,10,(n,2)); f=fitness(p); k=np.argmax(f)
    st.session_state.update(p=p,iteration=0,history=[float(f[k])],best=p[k].copy(),running=False,config=(n,scenario),previous=p.copy())
if st.session_state.get('config')!=(n,scenario): reset()

def step():
    old=st.session_state.p.copy(); p=old.copy(); f=fitness(old); rng=np.random.default_rng()
    for i in range(n):
        for j in range(n):
            if f[j]>f[i]:
                delta=old[j]-p[i]; r2=np.sum((delta/10)**2)
                p[i]+=beta*np.exp(-gamma*r2)*delta
        p[i]+=alpha*2*(rng.random(2)-0.5)
    p=np.clip(p,0,10); scores=fitness(p); k=np.argmax(scores)
    st.session_state.previous=old; st.session_state.p=p; st.session_state.iteration+=1
    if scores[k]>fitness(st.session_state.best): st.session_state.best=p[k].copy()
    st.session_state.history.append(float(fitness(st.session_state.best)))

c1,c2,c3,c4=st.columns(4)
if c1.button('▶ Start',use_container_width=True): st.session_state.running=True
if c2.button('⏸ Pause',use_container_width=True): st.session_state.running=False
if c3.button('🐝 Next iteration',use_container_width=True,disabled=st.session_state.iteration>=limit):
    st.session_state.running=False; step()
if c4.button('↻ Reset garden',use_container_width=True): reset()
if st.session_state.running and st.session_state.iteration<limit: step()
if st.session_state.iteration>=limit: st.session_state.running=False
p=st.session_state.p; f=fitness(p); best=float(fitness(st.session_state.best))
a,b,c=st.columns(3); a.metric('🌙 Iteration',st.session_state.iteration); b.metric('✨ Best fitness',f'{best:.3f}'); c.metric('🌼 Fireflies',n)
left,right=st.columns([1.65,1])
with left:
    st.subheader('🌿 Explore the garden')
    x=np.linspace(0,10,100); X,Y=np.meshgrid(x,x); Z=fitness(np.stack([X,Y],axis=-1))
    fig=go.Figure(go.Heatmap(x=x,y=x,z=Z,colorscale=[[0,'#10272d'],[.5,'#345746'],[1,'#a5b96a']],showscale=False,hoverinfo='skip'))
    for old,new in zip(st.session_state.previous,p):
        fig.add_trace(go.Scatter(x=[old[0],new[0]],y=[old[1],new[1]],mode='lines',line=dict(color='rgba(220,245,160,.28)',width=1),hoverinfo='skip',showlegend=False))
    for scale,opacity in [(38,.07),(25,.15),(14,.35)]:
        fig.add_trace(go.Scatter(x=p[:,0],y=p[:,1],mode='markers',marker=dict(size=scale*(.4+.6*f),color='#f4ff9c',opacity=opacity),hoverinfo='skip',showlegend=False))
    fig.add_trace(go.Scatter(x=p[:,0],y=p[:,1],mode='markers+text',text=['✦']*n,textposition='middle center',textfont=dict(size=13,color='#fffbd0'),marker=dict(size=9+8*f,color=f,colorscale=[[0,'#78966e'],[1,'#faff9e']],cmin=0,cmax=1),customdata=f,hovertemplate='Fitness: %{customdata:.3f}<extra>Firefly</extra>',showlegend=False))
    fig.add_trace(go.Scatter(x=[st.session_state.best[0]],y=[st.session_state.best[1]],mode='markers',marker=dict(symbol='circle-open',size=28,color='white',line=dict(width=2)),name='Best found'))
    fig.update_layout(height=490,margin=dict(l=12,r=12,t=12,b=12),paper_bgcolor='#192f32',plot_bgcolor='#192f32',font_color='#e5efcf',xaxis=dict(range=[0,10],title='Position X',showgrid=False),yaxis=dict(range=[0,10],title='Position Y',scaleanchor='x',showgrid=False),showlegend=False)
    st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    st.caption('✦ Firefly • white ring = best location found so far • brighter background = higher objective value')
with right:
    st.subheader('📈 Glow progress')
    h=go.Figure(go.Scatter(y=st.session_state.history,mode='lines+markers',line=dict(color='#e6f59c',width=3),fill='tozeroy'))
    h.update_layout(height=260,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor='#192f32',plot_bgcolor='#192f32',font_color='#e5efcf',xaxis_title='Iteration',yaxis_title='Best fitness',yaxis_range=[0,1.05])
    st.plotly_chart(h,use_container_width=True,config={'displayModeBar':False})
    if scenario=='Strawberry stall location':
        st.markdown('### 🍓 Strawberry stall mission')
        st.write('Find a stall location with the highest estimated customer demand. Two promising areas compete for attention.')
        st.metric('Estimated customers/day (toy model)',round(best*100))
    else:
        st.markdown('### 🌙 Garden mission')
        st.write('Find the brightest region of the garden. Each firefly represents a possible position.')
    st.progress(min(1.,best),text='Best fitness relative to known model maximum')
    if best>=.98: st.success('✨ Lovely! A near-optimal location has been found.')
with st.expander('📖 Learning corner — How does it work?',expanded=True):
    st.markdown('**Brightness:** solution quality. **Attraction:** movement towards better candidates. **Distance:** distant candidates exert weaker attraction when γ > 0. **Randomness:** exploration of new positions.')
    st.latex(r'\beta(r)=\beta_0 e^{-\gamma r^2}')
    st.write('This teaching variant uses synchronous brightness comparisons, normalized distance, bounded positions and a random perturbation per firefly per iteration. The best-so-far solution is retained separately.')
    st.markdown('**Try it:** Reset the garden, set α to zero, and compare the results with α = 0.5. Does more exploration always help?')
    st.markdown('**Reading:** [Yang (2010), Firefly Algorithms for Multimodal Optimization](https://arxiv.org/abs/1003.1466) · [Fister et al. (2013), A comprehensive review of firefly algorithms](https://doi.org/10.1016/j.swevo.2013.06.001)')
st.caption('Made for CE learning • Synthetic objective functions, not real customer data • No guarantee of global optimality')
if st.session_state.running:
    time.sleep(speed); st.rerun()
