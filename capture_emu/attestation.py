"""capture_emu.attestation - emulated Android key attestation, emulation profile v1.

EMULATION: this is not a real Android attestation. A test-root X.509 chain root -> batch -> leaf is built with
``cryptography``; the leaf's subjectPublicKeyInfo is the device P-256 key and its extension OID
1.3.6.1.4.1.11129.2.1.17 carries a DER ``KeyDescription`` (``asn1crypto``) whose attestationChallenge is the enrolment
challenge. The chain is deterministic: every byte is a function of (challenge, pk_D) (deterministic ECDSA, fixed
validity dates and serials).

Interface:

- ``key_description(challenge) -> (DER, app_id DER)``.
- ``build(challenge, leaf_pub) -> (ders [leaf, batch, root], kd_der, app_id, root_k, batch_k)``.
- ``verify(ders, challenge, pk_x, pk_y) -> True`` (asserts).
- ``build_chain(pk64, challenge) -> (leaf_der, batch_der, root_der)``.
- ``root_der() -> bytes`` and ``ROOT_SHA256()`` (the pinned emulation root's DER SHA-256, ``1133a858...``).
- ``verify_chain(ders, pinned_root_sha, expected) -> None | reason``: every enrolment verifier check, returning a
  short reason string on the first failure (no exceptions): chain of three, issuer/subject names and signatures up to
  the root whose DER SHA-256 equals ``pinned_root_sha``; the extension present, decoding strictly with DER re-encoding
  equal to the stored bytes; attestation and KeyMint versions 300; both security levels StrongBox; rootOfTrust
  Verified and deviceLocked; the application id naming ``cutline.capture.emu``; the leaf key's H(pk_x || pk_y) equal to
  ``expected["pk_hash"]`` (and to ``expected["pk64"]`` if given); attestationChallenge equal to
  ``expected["challenge"]``.
- ``att_hash(ders)`` = H(leaf || batch || root) (= ``cutline.ledger.abi.att_hash``).
"""
import datetime
import hashlib

from asn1crypto.core import Boolean, Enumerated, Integer, Null, OctetString, Sequence, SetOf
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID, ObjectIdentifier

N_P256 = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
H = lambda b: hashlib.sha256(b).digest()  # noqa: E731
ATT_OID = ObjectIdentifier("1.3.6.1.4.1.11129.2.1.17")

# ------------------------------------------------------------------------------ ASN.1 subset (AOSP schema, version 300)
class SecurityLevel(Enumerated):
    _map = {0: "software", 1: "trusted_environment", 2: "strongbox"}


class VerifiedBootState(Enumerated):
    _map = {0: "verified", 1: "self_signed", 2: "unverified", 3: "failed"}


class RootOfTrust(Sequence):
    _fields = [("verified_boot_key", OctetString), ("device_locked", Boolean),
               ("verified_boot_state", VerifiedBootState), ("verified_boot_hash", OctetString)]


class SetOfInteger(SetOf):
    _child_spec = Integer


class AuthorizationList(Sequence):     # only the tags this profile uses, in the schema's order
    _fields = [
        ("purpose", SetOfInteger, {"explicit": 1, "optional": True}),
        ("algorithm", Integer, {"explicit": 2, "optional": True}),
        ("key_size", Integer, {"explicit": 3, "optional": True}),
        ("digest", SetOfInteger, {"explicit": 5, "optional": True}),
        ("ec_curve", Integer, {"explicit": 10, "optional": True}),
        ("no_auth_required", Null, {"explicit": 503, "optional": True}),
        ("creation_date_time", Integer, {"explicit": 701, "optional": True}),
        ("origin", Integer, {"explicit": 702, "optional": True}),
        ("root_of_trust", RootOfTrust, {"explicit": 704, "optional": True}),
        ("attestation_application_id", OctetString, {"explicit": 709, "optional": True}),
    ]


class KeyDescription(Sequence):
    _fields = [("attestation_version", Integer), ("attestation_security_level", SecurityLevel),
               ("keymint_version", Integer), ("keymint_security_level", SecurityLevel),
               ("attestation_challenge", OctetString), ("unique_id", OctetString),
               ("software_enforced", AuthorizationList), ("hardware_enforced", AuthorizationList)]


class AttestationPackageInfo(Sequence):
    _fields = [("package_name", OctetString), ("version", Integer)]


class SetOfPackageInfo(SetOf):
    _child_spec = AttestationPackageInfo


class SetOfOctetString(SetOf):
    _child_spec = OctetString


class AttestationApplicationId(Sequence):
    _fields = [("package_infos", SetOfPackageInfo), ("signature_digests", SetOfOctetString)]


# ------------------------------------------------------------------------------ emulation profile v1 constants
PROFILE = {
    "attestation_version": 300, "keymint_version": 300, "security_level": 2,           # StrongBox
    "creation_date_time_ms": 1767225600000,                                             # 2026-01-01T00:00:00Z
    "package_name": "cutline.capture.emu", "package_version": 1,
    "signing_cert_digest_label": "CUTLINE-emu-app-signing-cert-v1",
    "purpose": 2, "algorithm": 3, "key_size": 256, "digest": 4, "ec_curve": 1, "origin": 0,
    "root_key_label": "CUTLINE-emu-att-root-v1", "batch_key_label": "CUTLINE-emu-att-batch-v1",
    "root_cn": "CUTLINE Emulated Attestation Root v1", "batch_cn": "CUTLINE Emulated Attestation Batch Key v1",
    "leaf_cn": "Android Keystore Key",
    "serial": {"root": 3, "batch": 2, "leaf": 1},
    "not_before": "2026-01-01T00:00:00Z", "not_after": "2046-01-01T00:00:00Z",
}


def p256_from_label(label):
    d = int.from_bytes(H(label.encode()), "big") % (N_P256 - 1) + 1
    return ec.derive_private_key(d, ec.SECP256R1())


def key_description(challenge):
    app_id = AttestationApplicationId({
        "package_infos": [{"package_name": PROFILE["package_name"].encode(), "version": PROFILE["package_version"]}],
        "signature_digests": [H(PROFILE["signing_cert_digest_label"].encode())],
    }).dump()
    sw = AuthorizationList({"creation_date_time": PROFILE["creation_date_time_ms"], "attestation_application_id": app_id})
    hw = AuthorizationList({
        "purpose": [PROFILE["purpose"]], "algorithm": PROFILE["algorithm"], "key_size": PROFILE["key_size"],
        "digest": [PROFILE["digest"]], "ec_curve": PROFILE["ec_curve"], "no_auth_required": Null(),
        "origin": PROFILE["origin"],
        "root_of_trust": {"verified_boot_key": bytes(32), "device_locked": True, "verified_boot_state": "verified",
                          "verified_boot_hash": bytes(32)},
    })
    kd = KeyDescription({
        "attestation_version": PROFILE["attestation_version"], "attestation_security_level": "strongbox",
        "keymint_version": PROFILE["keymint_version"], "keymint_security_level": "strongbox",
        "attestation_challenge": challenge, "unique_id": b"", "software_enforced": sw, "hardware_enforced": hw,
    })
    return kd.dump(force=True), app_id


def ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)


def name(cn):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def build(challenge, leaf_pub):
    root_k, batch_k = p256_from_label(PROFILE["root_key_label"]), p256_from_label(PROFILE["batch_key_label"])
    nb, na = ts(PROFILE["not_before"]), ts(PROFILE["not_after"])

    def ku(**kw):
        base = dict(digital_signature=False, content_commitment=False, key_encipherment=False, data_encipherment=False,
                    key_agreement=False, key_cert_sign=False, crl_sign=False, encipher_only=False, decipher_only=False)
        base.update(kw)
        return x509.KeyUsage(**base)

    root = (x509.CertificateBuilder().subject_name(name(PROFILE["root_cn"])).issuer_name(name(PROFILE["root_cn"]))
            .public_key(root_k.public_key()).serial_number(PROFILE["serial"]["root"])
            .not_valid_before(nb).not_valid_after(na)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(ku(key_cert_sign=True, crl_sign=True), critical=True)
            .sign(root_k, hashes.SHA256(), ecdsa_deterministic=True))
    batch = (x509.CertificateBuilder().subject_name(name(PROFILE["batch_cn"])).issuer_name(name(PROFILE["root_cn"]))
             .public_key(batch_k.public_key()).serial_number(PROFILE["serial"]["batch"])
             .not_valid_before(nb).not_valid_after(na)
             .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
             .add_extension(ku(key_cert_sign=True), critical=True)
             .sign(root_k, hashes.SHA256(), ecdsa_deterministic=True))
    kd_der, app_id = key_description(challenge)
    leaf = (x509.CertificateBuilder().subject_name(name(PROFILE["leaf_cn"])).issuer_name(name(PROFILE["batch_cn"]))
            .public_key(leaf_pub).serial_number(PROFILE["serial"]["leaf"])
            .not_valid_before(nb).not_valid_after(na)
            .add_extension(ku(digital_signature=True), critical=True)
            .add_extension(x509.UnrecognizedExtension(ATT_OID, kd_der), critical=False)
            .sign(batch_k, hashes.SHA256(), ecdsa_deterministic=True))
    ders = [c.public_bytes(serialization.Encoding.DER) for c in (leaf, batch, root)]
    return ders, kd_der, app_id, root_k, batch_k


def verify(ders, challenge, pk_x, pk_y):
    leaf, batch, root = (x509.load_der_x509_certificate(d) for d in ders)
    for child, parent in ((leaf, batch), (batch, root), (root, root)):
        assert child.issuer == parent.subject
        child.verify_directly_issued_by(parent)
    pn = leaf.public_key().public_numbers()
    assert (pn.x.to_bytes(32, "big"), pn.y.to_bytes(32, "big")) == (pk_x, pk_y)
    ext = leaf.extensions.get_extension_for_oid(ATT_OID)
    kd = KeyDescription.load(ext.value.value, strict=True)
    assert kd["attestation_challenge"].native == challenge
    assert kd["attestation_security_level"].native == "strongbox" and kd["keymint_security_level"].native == "strongbox"
    assert kd["attestation_version"].native == 300 and kd["keymint_version"].native == 300
    hw, sw = kd["hardware_enforced"], kd["software_enforced"]
    assert hw["root_of_trust"]["verified_boot_state"].native == "verified" and hw["root_of_trust"]["device_locked"].native
    assert hw["ec_curve"].native == 1 and hw["algorithm"].native == 3 and hw["origin"].native == 0
    app = AttestationApplicationId.load(sw["attestation_application_id"].native, strict=True)
    assert app["package_infos"][0]["package_name"].native == PROFILE["package_name"].encode()
    assert kd.dump(force=True) == ext.value.value            # the extension is DER (re-encoding is identity)
    return True


# ------------------------------------------------------------------------------ wrappers used by the app and the enrolment check
def _pub_from64(pk64):
    pk64 = bytes(pk64)
    if len(pk64) != 64:
        raise ValueError("pk must be 64 bytes pk_x || pk_y")
    return ec.EllipticCurvePublicNumbers(int.from_bytes(pk64[:32], "big"), int.from_bytes(pk64[32:], "big"),
                                         ec.SECP256R1()).public_key()


def build_chain(pk64, challenge):
    ders, _, _, _, _ = build(bytes(challenge), _pub_from64(pk64))
    return tuple(ders)


_ROOT = {}


def root_der():
    """The emulation root's DER (independent of the challenge and the leaf key: built once with a placeholder leaf)."""
    if "der" not in _ROOT:
        placeholder = p256_from_label(PROFILE["root_key_label"]).public_key()
        _ROOT["der"] = build(b"\x00" * 32, placeholder)[0][2]
    return _ROOT["der"]


def ROOT_SHA256():
    return H(root_der())


def att_hash(ders):
    return H(b"".join(bytes(d) for d in ders))


def verify_chain(ders, pinned_root_sha, expected):
    """None if the chain passes every enrolment check, else a short reason (the enrolment is refused)."""
    try:
        ders = [bytes(d) for d in ders]
        if len(ders) != 3:
            return "chain length"
        if H(ders[2]) != bytes(pinned_root_sha):
            return "root not pinned"
        try:
            leaf, batch, root = (x509.load_der_x509_certificate(d) for d in ders)
        except ValueError:
            return "certificate DER"
        for child, parent in ((leaf, batch), (batch, root), (root, root)):
            if child.issuer != parent.subject:
                return "issuer name"
            try:
                child.verify_directly_issued_by(parent)
            except Exception:
                return "certificate signature"
        try:
            ext = leaf.extensions.get_extension_for_oid(ATT_OID)
        except x509.ExtensionNotFound:
            return "no attestation extension"
        raw = ext.value.value
        try:
            kd = KeyDescription.load(raw, strict=True)
            if kd.dump(force=True) != raw:
                return "extension not DER"
            kd.native  # noqa: B018  (forces a full parse)
        except Exception:
            return "extension decode"
        if kd["attestation_version"].native != PROFILE["attestation_version"] or \
                kd["keymint_version"].native != PROFILE["keymint_version"]:
            return "version"
        if kd["attestation_security_level"].native != "strongbox" or kd["keymint_security_level"].native != "strongbox":
            return "security level"
        hw, sw = kd["hardware_enforced"], kd["software_enforced"]
        rot = hw["root_of_trust"]
        if rot.native is None or rot["verified_boot_state"].native != "verified" or not rot["device_locked"].native:
            return "root of trust"
        if hw["ec_curve"].native != PROFILE["ec_curve"] or hw["algorithm"].native != PROFILE["algorithm"] or \
                hw["origin"].native != PROFILE["origin"]:
            return "key parameters"
        try:
            app = AttestationApplicationId.load(sw["attestation_application_id"].native, strict=True)
            names = [p["package_name"].native for p in app["package_infos"]]
        except Exception:
            return "application id"
        if PROFILE["package_name"].encode() not in names:
            return "application id"
        pn = leaf.public_key().public_numbers()
        pk64 = pn.x.to_bytes(32, "big") + pn.y.to_bytes(32, "big")
        if "pk64" in expected and pk64 != bytes(expected["pk64"]):
            return "leaf key"
        if H(pk64) != bytes(expected["pk_hash"]):
            return "pkHash"
        if kd["attestation_challenge"].native != bytes(expected["challenge"]):
            return "challenge"
        return None
    except Exception as e:  # any parsing surprise refuses the enrolment
        return f"malformed: {type(e).__name__}"
