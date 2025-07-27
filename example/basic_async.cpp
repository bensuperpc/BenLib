#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/rsa.h>
#include <openssl/kdf.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

std::string b64Encode(const std::vector<unsigned char>& str) {
    BIO *b64 = BIO_new(BIO_f_base64()), *mem = BIO_new(BIO_s_mem());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    mem = BIO_push(b64, mem);
    if (BIO_write(mem, str.data(), static_cast<int>(str.size())) <= 0) {
        BIO_free_all(mem);
        return "";
    }
    BIO_flush(mem);
    BUF_MEM* bptr = nullptr;
    BIO_get_mem_ptr(mem, &bptr);
    std::string out(bptr->data, bptr->length);
    BIO_free_all(mem);
    return out;
}

std::vector<unsigned char> b64Decode(const std::string& str) {
    BIO *b64 = BIO_new(BIO_f_base64()), *mem = BIO_new_mem_buf(str.data(), static_cast<int>(str.size()));
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    mem = BIO_push(b64, mem);
    std::vector<unsigned char> out(str.size());
    int len = BIO_read(mem, out.data(), static_cast<int>(out.size()));
    out.resize(std::max(0, len));
    BIO_free_all(mem);
    return out;
}

bool deriveAESKey(const std::vector<unsigned char>& secret, const std::vector<unsigned char>& salt, unsigned char* aes_key, size_t aes_key_len) {
    if (aes_key_len != 32 && aes_key_len != 16 && aes_key_len != 24) {
        std::cerr << "Invalid AES key length: " << aes_key_len << ". Must be 16, 24, or 32 bytes." << std::endl;
        return false;
    }
    if (secret.empty() || salt.empty() || aes_key == nullptr) {
        return false;
    }

    EVP_KDF* kdf = EVP_KDF_fetch(nullptr, "HKDF", nullptr);
    if (!kdf) {
        return false;
    }

    EVP_KDF_CTX* kctx = EVP_KDF_CTX_new(kdf);
    EVP_KDF_free(kdf);
    if (!kctx) {
        return false;
    }

    const char context_info[] = "aes encryption key";
    OSSL_PARAM params[] = {
        OSSL_PARAM_construct_utf8_string("digest", const_cast<char*>("SHA3-512"), 0),
        OSSL_PARAM_construct_octet_string("salt", const_cast<unsigned char*>(salt.data()), salt.size()),
        OSSL_PARAM_construct_octet_string("key", const_cast<unsigned char*>(secret.data()), secret.size()),
        OSSL_PARAM_construct_octet_string("info", const_cast<char*>(context_info), strlen(context_info)),
        OSSL_PARAM_construct_end()
    };

    int res = EVP_KDF_derive(kctx, aes_key, aes_key_len, params);
    EVP_KDF_CTX_free(kctx);

    return res == 1;
}

bool AESEncrypt(const std::string& plaintext,
        const unsigned char key[32],
        std::vector<unsigned char>& ivOutput,
        std::vector<unsigned char>& ciphertext,
        std::vector<unsigned char>& tagOutput,
        const EVP_CIPHER* cipher = EVP_aes_256_gcm()) {
    ivOutput.resize(12);
    if (RAND_bytes(ivOutput.data(), 12) != 1) {
        return false;
    }

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx)
        return false;

    if (EVP_EncryptInit_ex(ctx, cipher, nullptr, nullptr, nullptr) != 1 || EVP_EncryptInit_ex(ctx, nullptr, nullptr, key, ivOutput.data()) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }

    int len = 0;
    if (EVP_EncryptUpdate(ctx, nullptr, &len, nullptr, 0) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }

    ciphertext.resize(plaintext.size());
    if (EVP_EncryptUpdate(ctx, ciphertext.data(), &len, reinterpret_cast<const unsigned char*>(plaintext.data()), static_cast<int>(plaintext.size())) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }
    int cipher_len = len;

    if (EVP_EncryptFinal_ex(ctx, ciphertext.data() + cipher_len, &len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }
    cipher_len += len;
    ciphertext.resize(static_cast<size_t>(cipher_len));
    tagOutput.resize(16);
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tagOutput.data()) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }

    EVP_CIPHER_CTX_free(ctx);
    return true;
}

bool AESDecrypt(std::string& plaintext_out,
        const unsigned char key[32],
        const std::vector<unsigned char>& iv,
        const std::vector<unsigned char>& tag,
        const std::vector<unsigned char>& ciphertext,
        size_t cipher_len,
        const EVP_CIPHER* cipher = EVP_aes_256_gcm()) {
    if (iv.size() != 12 || tag.size() != 16 || ciphertext.empty() || cipher_len == 0) {
        return false;
    }

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        return false;
    }

    if (EVP_DecryptInit_ex(ctx, cipher, nullptr, nullptr, nullptr) != 1 || EVP_DecryptInit_ex(ctx, nullptr, nullptr, key, iv.data()) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }

    int len = 0;
    if (EVP_DecryptUpdate(ctx, nullptr, &len, nullptr, 0) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }

    std::vector<unsigned char> buf(cipher_len);
    if (EVP_DecryptUpdate(ctx, buf.data(), &len, ciphertext.data(), static_cast<int>(cipher_len)) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }
    int pt_len = len;

    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, 16, const_cast<unsigned char*>(tag.data())) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }

    if (EVP_DecryptFinal_ex(ctx, buf.data() + pt_len, &len) <= 0) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }
    pt_len += len;

    plaintext_out.assign(reinterpret_cast<char*>(buf.data()), static_cast<size_t>(pt_len));
    OPENSSL_cleanse(buf.data(), buf.size());

    EVP_CIPHER_CTX_free(ctx);
    return true;
}

bool writePKey(const EVP_PKEY* pkey, const std::filesystem::path& path, bool pub) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    
    bool res = false;
    if (pub) {
        res =  PEM_write_PUBKEY(f, pkey);
    } else {
        res = PEM_write_PrivateKey(f, pkey, nullptr, nullptr, 0, nullptr, nullptr);
    }
    fclose(f);
    return res;
}

EVP_PKEY* readPKey(const std::filesystem::path& path, bool pub) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        return nullptr;
    }

    EVP_PKEY* pkey = nullptr;

    if (pub) {
        pkey = PEM_read_PUBKEY(f, nullptr, nullptr, nullptr);
    } else {
       pkey = PEM_read_PrivateKey(f, nullptr, nullptr, nullptr);
    }
    fclose(f);
    return pkey;
}

bool generateKeypair(const std::filesystem::path& pub_path, const std::filesystem::path& priv_path, int keyType = EVP_PKEY_X448) {
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(keyType, nullptr);
    if (!ctx) {
        return false;
    }
    if (EVP_PKEY_keygen_init(ctx) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return false;
    }
    EVP_PKEY* pkey = nullptr;
    if (EVP_PKEY_keygen(ctx, &pkey) <= 0) {
        EVP_PKEY_CTX_free(ctx);
        return false;
    }
    EVP_PKEY_CTX_free(ctx);
    bool ok = writePKey(pkey, pub_path, true) && writePKey(pkey, priv_path, false);
    EVP_PKEY_free(pkey);
    return ok;
}

std::string encryptMessage(EVP_PKEY* publicKey, const std::string& message) {
    if (!publicKey) {
        return "";
    }
    if (message.empty()) {
        return "";
    }

    EVP_PKEY_CTX* kctx = EVP_PKEY_CTX_new(publicKey, nullptr);

    EVP_PKEY* eph = nullptr;
    if (!kctx || EVP_PKEY_keygen_init(kctx) <= 0 || EVP_PKEY_keygen(kctx, &eph) <= 0) {
        if (kctx) EVP_PKEY_CTX_free(kctx);
        return "";
    }
    EVP_PKEY_CTX_free(kctx);

    EVP_PKEY_CTX* dctx = EVP_PKEY_CTX_new(eph, nullptr);
    if (!dctx || EVP_PKEY_derive_init(dctx) <= 0 || EVP_PKEY_derive_set_peer(dctx, publicKey) <= 0) {
        EVP_PKEY_free(eph);
        EVP_PKEY_CTX_free(dctx);
        return "";
    }
    
    size_t secret_len = 0;
    if (EVP_PKEY_derive(dctx, nullptr, &secret_len) <= 0) {
        EVP_PKEY_free(eph);
        EVP_PKEY_CTX_free(dctx);
        return "";
    }
    std::vector<unsigned char> secret(secret_len);
    if (EVP_PKEY_derive(dctx, secret.data(), &secret_len) <= 0) {
        EVP_PKEY_free(eph);
        EVP_PKEY_CTX_free(dctx);
        return "";
    }
    EVP_PKEY_CTX_free(dctx);

    std::vector<unsigned char> salt(32);
    if (RAND_bytes(salt.data(), salt.size()) != 1)
        return "";

    unsigned char* aes_key = static_cast<unsigned char*>(OPENSSL_secure_malloc(32));

    if (!deriveAESKey(secret, salt, aes_key, 32)) {
        OPENSSL_secure_free(aes_key);
        return "";
    }

    std::vector<unsigned char> iv = {};
    std::vector<unsigned char> ct = {};
    std::vector<unsigned char> tag = {};
    if (!AESEncrypt(message, aes_key, iv, ct, tag)) {
        EVP_PKEY_free(eph);
        return "";
    }

    OPENSSL_cleanse(aes_key, 32);
    OPENSSL_secure_free(aes_key);

    BIO* bio = BIO_new(BIO_s_mem());
    if (PEM_write_bio_PUBKEY(bio, eph) <= 0) {
        EVP_PKEY_free(eph);
        return "";
    }
    BUF_MEM* bptr = nullptr;
    BIO_get_mem_ptr(bio, &bptr);
    std::vector<unsigned char> ephpub(bptr->data, bptr->data + bptr->length);
    BIO_free_all(bio);
    EVP_PKEY_free(eph);

    std::vector<unsigned char> out = {};
    out.insert(out.end(), ephpub.begin(), ephpub.end());
    out.insert(out.end(), salt.begin(), salt.end());
    out.insert(out.end(), iv.begin(), iv.end());
    out.insert(out.end(), tag.begin(), tag.end());
    out.insert(out.end(), ct.begin(), ct.end());

    return b64Encode(out);
}

std::string decryptMessage(EVP_PKEY* privateKey, const std::string& encrypted_base64) {
    if (!privateKey) {
        return "";
    }
    if (encrypted_base64.empty()) {
        return "";
    }
    std::vector<unsigned char> bin = b64Decode(encrypted_base64);

    const std::string end_marker = "\n-----END PUBLIC KEY-----\n";
    auto it = std::search(bin.begin(), bin.end(), end_marker.begin(), end_marker.end());
    if (it == bin.end()) {
        return "";
    }
    
    it += end_marker.size();
    size_t ephLengh = it - bin.begin();

    BIO* bio = BIO_new_mem_buf(bin.data(), static_cast<int>(ephLengh));
    if (!bio) {
        return "";
    }

    EVP_PKEY* eph = PEM_read_bio_PUBKEY(bio, nullptr, nullptr, nullptr);
    if (!eph) {
        BIO_free_all(bio);
        return "";
    }

    BIO_free_all(bio);
    if (!eph) {
        return "";
    }

    EVP_PKEY_CTX* dctx = EVP_PKEY_CTX_new(privateKey, nullptr);
    if (!dctx || EVP_PKEY_derive_init(dctx) <= 0 || EVP_PKEY_derive_set_peer(dctx, eph) <= 0) {
        EVP_PKEY_free(eph);
        return "";
    }
    size_t secret_len = 0;
    if (EVP_PKEY_derive(dctx, nullptr, &secret_len) <= 0) {
        EVP_PKEY_free(eph);
        EVP_PKEY_CTX_free(dctx);
        return "";
    }
    std::vector<unsigned char> secret(secret_len);
    if (EVP_PKEY_derive(dctx, secret.data(), &secret_len) <= 0) {
        EVP_PKEY_free(eph);
        EVP_PKEY_CTX_free(dctx);
        return "";
    }
    EVP_PKEY_CTX_free(dctx);
    EVP_PKEY_free(eph);

    unsigned char* aes_key = static_cast<unsigned char*>(OPENSSL_secure_malloc(32));

    size_t pos = ephLengh;
    const unsigned char* salt = bin.data() + pos;
    pos += 32;
    std::vector<unsigned char> iv = std::vector<unsigned char>(bin.data() + pos, bin.data() + pos + 12);
    pos += 12;
    std::vector<unsigned char> tag = std::vector<unsigned char>(bin.data() + pos, bin.data() + pos + 16);
    pos += 16;
    std::vector<unsigned char> ct = std::vector<unsigned char>(bin.data() + pos, bin.data() + bin.size());
    size_t ct_len = bin.size() - pos;

    if (!deriveAESKey(secret, std::vector<unsigned char>(salt, salt + 32), aes_key, 32)) {
        OPENSSL_secure_free(aes_key);
        return "";
    }

    std::string plaintext = "";
    bool ok = AESDecrypt(plaintext, aes_key, iv, tag, ct, ct_len);
    OPENSSL_cleanse(aes_key, 32);
    OPENSSL_secure_free(aes_key);

    if (!ok) {
        return "";
    }

    return plaintext;
}

int main() {
    if (OPENSSL_init_crypto(OPENSSL_INIT_LOAD_CONFIG, NULL) != 1) {
    }

    const std::string pub = "ec_public.pem";
    const std::string priv = "ec_private.pem";

    if (!generateKeypair(pub, priv)) {
        std::cerr << "Keypair generation failed\n";
        return 1;
    }

    std::string msg = "Hello, EC-GCM world!";

    EVP_PKEY* publicKey = readPKey(pub, true);
    std::string encrypted = encryptMessage(publicKey, msg);
    if (encrypted.empty()) {
        std::cerr << "Encryption failed\n";
        EVP_PKEY_free(publicKey);
        return 1;
    }
    EVP_PKEY_free(publicKey);

    EVP_PKEY* privateKey = readPKey(priv, false);
    std::string decrypted = decryptMessage(privateKey, encrypted);
    if (decrypted.empty()) {
        std::cerr << "Decryption failed\n";
        EVP_PKEY_free(publicKey);
        return 1;
    }
    EVP_PKEY_free(publicKey);

    std::cout << "Plaintext: " << msg << "\n"
              << "Encrypted: " << encrypted << "\n"
              << "Decrypted: " << decrypted << std::endl;

    if (msg != decrypted) {
        std::cerr << "Decrypted message does not match original\n";
        return 1;
    }
    std::cout << "Decryption successful!\n";

    std::filesystem::remove(pub);
    std::filesystem::remove(priv);
    OPENSSL_cleanup();
    std::cout << "Keys cleaned up.\n";
    return 0;
}
