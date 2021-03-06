all: vanity
clean: clean_credential
	$(RM) vanity
clean_credential:
	$(RM) input.sec result.sec input.pub result.pub private.key
	$(RM) -r gpg

OPENSSL_CFLAGS := `pkg-config --cflags openssl`
OPENSSL_LIBS := `pkg-config --libs openssl`

CUDAFLAGS= -O3 -rdc=true -gencode arch=compute_50,code=sm_50
NVCC=nvcc

# vanity: vanity.c
# 	$(CC) \
# 		-o $@ \
# 		-O3 \
# 		-W \
# 		-Wall \
# 		$< \
# 		$(OPENSSL_CFLAGS) \
# 		$(OPENSSL_LIBS) \

vanity: vanity_cuda.cu sha1.cu
	$(NVCC) $(CUDAFLAGS) vanity_cuda.cu sha1.cu -o vanity
