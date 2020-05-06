from pypai import PAI

# Create a PAI cluster
pai = PAI(username='zm', passwd='123456')

# Generate the configuration
pai.submit(exclude_file='.h5')