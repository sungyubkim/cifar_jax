import jax

def cross_replica_mean(replicated):
    return jax.pmap(lambda x: jax.lax.pmean(x,'x'),'x')(replicated)

def unreplicate(tree, i=0):
  """Returns a single instance of a replicated array."""
  return jax.tree_map(lambda x: x[i], tree)