"""Phase-aware resource evidence; importing this file never touches a driver.

Only snapshots from an already launched kernel or the Manager's explicit
loaded_resources record can establish observed SDK register/spill counts.
Compiler metadata is reported separately. No handle initialization, driver
attribute query, launch, synchronization or GPU import is performed here.
"""

META_FIELDS = ('num_warps','num_stages','shared','use_sme','warp_size')


def counts_known(record):
    return (type(record.get('n_regs')) is int and record['n_regs']>0
            and type(record.get('n_spills')) is int and record['n_spills']>=0)


def resource_evidence(snapshot, manager_record=None, *, already_launched=False):
    evidence=dict(hash=snapshot.get('hash'),counts_status='unknown',n_regs=None,n_spills=None,
        basis=None,post_launch_attribute_counts={'n_regs':snapshot.get('n_regs'),
                                                'n_spills':snapshot.get('n_spills')},
        manager_prelaunch_counts=None,manager_loaded_counts=None,
        matching_manager_binary_identity=False,driver_function_attributes_queried=False,
        driver_local_size_bytes=None,driver_static_shared_bytes=None,
        complete_resource_equivalence_claimed=False)
    observed=[]
    if already_launched and counts_known(snapshot):
        observed.append(('post_launch_kernel_attributes',snapshot))
    if manager_record is not None and 'loaded_resources' in manager_record:
        before=manager_record.get('candidate_resources',{})
        loaded=manager_record['loaded_resources']
        identity=manager_record.get('candidate_hash')
        if not (snapshot.get('hash')==before.get('hash')==loaded.get('hash')==identity
                and snapshot.get('metadata')==before.get('metadata')==loaded.get('metadata')
                and snapshot.get('asm_sha256')==before.get('asm_sha256')==loaded.get('asm_sha256')):
            raise ValueError('Manager pre/post/snapshot binary identity mismatch')
        evidence['matching_manager_binary_identity']=True
        evidence['manager_prelaunch_counts']={key:before.get(key) for key in ('n_regs','n_spills')}
        evidence['manager_loaded_counts']={key:loaded.get(key) for key in ('n_regs','n_spills')}
        if counts_known(loaded):observed.append(('manager_loaded_resources',loaded))
    if observed:
        values={(record['n_regs'],record['n_spills']) for _,record in observed}
        if len(values)!=1:raise ValueError('Conflicting post-launch SDK count evidence')
        regs,spills=values.pop()
        evidence.update(counts_status='observed_post_launch',n_regs=regs,n_spills=spills,
                        basis=[basis for basis,_ in observed])
    return evidence


def compare_resource_records(old,new):
    """Compare only observed fields. Unknown never compares equal to unknown."""
    def metadata_value_known(key,value):
        return type(value) is int and value>=(0 if key in ('shared','use_sme') else 1)
    metadata_unknown=[key for key in META_FIELDS
                      if not metadata_value_known(key,old.get('metadata',{}).get(key))
                      or not metadata_value_known(key,new.get('metadata',{}).get(key))]
    metadata_changed=[key for key in META_FIELDS if key not in metadata_unknown
                      and old['metadata'][key]!=new['metadata'][key]]
    left,right=(record.get('post_launch_resource_evidence',{}) for record in (old,new))
    known=counts_known(left) and counts_known(right)
    equal=((left['n_regs'],left['n_spills'])==(right['n_regs'],right['n_spills'])) if known else None
    changed=list(metadata_changed)
    if known:
        changed.extend(key for key in ('n_regs','n_spills') if left[key]!=right[key])
    meta_known=not metadata_unknown
    metadata_equal=(not metadata_changed) if meta_known else None
    strict=meta_known and metadata_equal and known and equal
    record=meta_known and metadata_equal and known
    return dict(known_metadata_fields=list(META_FIELDS),metadata_unknown=metadata_unknown,
        required_metadata_known=meta_known,known_metadata_equal=metadata_equal,
        observed_counts_known=known,same_registers_and_spills=equal,
        changed_observed_fields=changed,resource_changed=(True if changed else False if meta_known and known else None),
        all_required_observed_fields_known=meta_known and known,
        strict_allowed=bool(strict),record_allowed=bool(record),
        unqueried_driver_fields=['local_size_bytes','static_shared_bytes'],
        complete_resource_equivalence_claimed=False,
        timing_attribution='Full candidate including ordering, metadata/index work and any changed observed resource footprint; not a pure-CTA attribution.')
