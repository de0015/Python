# So in the code below we will be creating a blockchain using a few items. 
# The blockchain will contain the index of the block, the timestamp the block was created and the data contained within the block.
# Lastly the block will be hashed using the sha256 encryption algorthm and the next block will be compared against the hash of the previous one.

import hashlib as hasher

# We need to import the hashlib to create the hash of the data that is collected within the blockchain

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.hash.block()

    def hash_block(self):
        sha = hasher.sha256()
        sha.update(str(self.index) +
                   str(self.timestamp) +
                   str(self.data) +
                   str(self.previous_hash))
        return sha.hexdigest()
        
import datetime as date

def create_genisis_block():
    return Block(0, date.datetime.now(), "Genesis Block", "0")

def next_block(last_block):
    this_index = last_block.index + 1 
    this_timestamp = date.datetime.now()
    this_data = "Block" + str(this_index)
    this_hash = last_block.hash
    return Block(this_index, this_timestamp, this_data, this_hash)

# Here is where we create the blockchain and add the Genesis block!!

blockchain = [create_genisis_block]
previous_block = blockchain[0]

#now we choose how many blocks we want on the chain

num_of_blocks = 500

# now we get to add the blocks to the chain

for i in range(0, num_of_blocks):
    block_to_add = next_block(previous_block)
    blockchain.append(block_to_add)
    previous_block = block_to_add

    print "Block #{} has been added to the chain".format(block_to_add.index)
    print "Hash: {}\n".format(block_to_add.hash)