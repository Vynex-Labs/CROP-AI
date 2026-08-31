# Structured IPM knowledge

This directory stores **verified, structured** advisory records.

Rules (MASTER_PROMPT §28):

- Do not invent chemical dosage, frequency, pre-harvest interval, or legal approval.
- Cultural / mechanical / monitoring actions may be listed when they come from a cited official package.
- If a required chemical detail is missing, the advisory engine must escalate to extension / laboratory consultation.
- Generative models must not override these records.

Each crop file is YAML. `status: placeholder` means the record is a citation stub, not a complete official package.

Primary citation targets (to be filled when packages are attached by an agronomist):

- NIPHM IPM packages: https://niphm.gov.in/IPMPackages/
- Maharashtra CROPSAP advisories (authorised dump, not redistributed here)
- State agriculture university recommendations (MPKV / VNMKV / Dr. PDKV / BSKKV)
- CIBRC approved labels for any chemical product names

Until a record has `status: verified` and a `source_url` or attached PDF hash, the decision engine may only emit **monitoring + expert referral**.
