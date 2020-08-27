all: vanity
clean: clean_credential
	$(RM) vanity
clean_credential:
	$(RM) input.sec result.sec input.pub result.pub private.key
	$(RM) -r gpg

OPENSSL_CFLAGS := `pkg-config --cflags openssl`
OPENSSL_LIBS := `pkg-config --libs openssl`

vanity: vanity.c
	$(CC) \
		-o $@ \
		-O3 \
		-W \
		-Wall \
		$< \
		$(OPENSSL_CFLAGS) \
		$(OPENSSL_LIBS) \
