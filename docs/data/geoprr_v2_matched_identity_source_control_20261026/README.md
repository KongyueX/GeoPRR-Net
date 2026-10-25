# Matched identity15E source控制

**科学结论NO_MATCHED_SUPPORT_FOR_C3：没有满足五项同时成立的预注册匹配支持。** C3的R_view0.06737849594632048低于identity0.07481353829650483，mean／tail drift也更低；但两视图>5错误总数C3为59、identity33，最大误差C3为0.5509077310562134、identity0.5424728989601135，两项未满足。两臂native／CF旧门槛通过与matched主判定否定分开，不因风险或漂移较好启动C3长训、promotion或扫描重试。

[完整报告](completed/report_cn.md)、[匹配判定](completed/matched_comparison.json)、[指标](completed/metrics.json)及[最终审计](completed/numeric_execution_audit.json)保留混合证据。Identity native mean／tail／max2.543121824／11.800955331／53.797036409%FS，>1／2／5为190／110／16；identity CF2.635376286／12.945961876／54.247289896%FS，188／103／17。C3 native2.583311489／11.098201560／55.090773106，189／120／28；C3 CF2.558655712／10.711229617／53.595584631，192／120／31。Identity最大prediction漂移30.185824633%FS亦保留，未称任一模型完全color invariant。

[冻结协议](../../GEOPRR_V2_MATCHED_IDENTITY_SOURCE_CONTROL_CN.md)只删除训练通道变化，15E两臂均82440训练呈现／1290尝试，实际均1289更新＋1AMPskip；model／factory／cache／seed、source5496／687parent、ID292／73layout、RGB512／原photo／SmoothL1及其余优化设置按保存代码／metadata匹配。Identity每E实际RGB5496／GBR0／BRG0，history[1832,1832,1832]只为nominal slots，全部映射RGB，不能误写C3三颜色训练。

完整identity native／constant／CF三份scalar CSV.gz及73组summary／CSV、两份C3→identity292行pairs及group对比、15E history／gates／matched comparison保留；C3旧模型不再forward。正式83024图／1300 CNN及point调用，pref另8unique／3CNN／1back／0optimizer，native identity额外4ID decode计入wall。Source03／Industrial／RF全部0RGB／模型，无真实域或新盲测结果。

[实际成本](completed/costs.json)保留train446.680151045秒／native＋CF pipeline6.870805331秒，formal matched-wrapper full460.987576280秒、whole supervisor482.665441178秒；supervisor pref／control19.270640597／462.370882876秒。Point／operator独立时间unknown／null，不用wall差估净算子成本；wrapper metadata／reference检查与pref额外decode包含在对应entry／wall。Cached／BN／strict等只保存见证，未新开旧weights内容。

最终provided-label审计171 PASS／0 FAIL、13726 numeric comparisons、maxdiff1.4210854715202004e-14；参考Acc@5末位roundoff在既有1e-12容差内接受，非bitwise一致。[Reference资格](completed/reference_audit_qualification.json)及报告Limitations保留raw-label派生范围，旧C3审计149／148／1和原FAIL不变，不写成旧C3 clean PASS。根目录7项＋独立协议1项＋[completed31项](completed/public_allowlist.json)，[总显式清单](public_allowlist.json)39项；fullcontracts／训练arrays／RGB／weights／features／geometry／privatepaths不公开，旧目录不改。
