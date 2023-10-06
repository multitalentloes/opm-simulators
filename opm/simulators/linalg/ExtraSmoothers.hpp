#ifndef OPM_EXTRASMOOTHERS_HPP
#define OPM_EXTRASMOOTHERS_HPP

#include "cuistl/CuDILU.cpp"

namespace Dune
{
  template <class M, class X, class Y>
    class SeqDilu;

namespace Amg
{

    template <class T>
    class ConstructionTraits;

    template <class T>
    struct SmootherTraits;


    template <class F>
    struct DiluSmootherArgs : public Dune::Amg::DefaultSmootherArgs<F> {
        //bool leftPrecond;
        //DiluSmootherArgs();
	  //:
	  //iterations(1),
	  //relaxationFactor(1.0),
	  //leftPrecond(true)
    //   {
    //   }
    };
  template <class M, class X, class Y>
  struct SmootherTraits< Dune::SeqDilu<M, X, Y>> {
    //typedef DiluSmootherArgs< Dune::SeqDilu<M, X, Y> > Arguments;
   typedef DiluSmootherArgs< double > Arguments;
  };


    /**
     * @brief Policy for the construction of the SeqDilu smoother
     */
    template <class M, class X, class Y>
    struct ConstructionTraits<SeqDilu<M, X, Y>> {
        typedef DefaultConstructionArgs<SeqDilu<M, X, Y>> Arguments;
#if DUNE_VERSION_NEWER(DUNE_ISTL, 2, 7)
        static inline std::shared_ptr<SeqDilu<M, X, Y>> construct(Arguments& args)
        {
            return std::make_shared<SeqDilu<M, X, Y>>(
							  //args.getMatrix(), args.getArgs().iterations, args.getArgs().relaxationFactor,args.getArgs().leftPrecond);
							  args.getMatrix());
        }

#else
        static inline SeqDilu<M, X, Y>* construct(Arguments& args)
        {
            return new SeqDilu<M, X, Y>(
					    //args.getMatrix(), args.getArgs().iterations, args.getArgs().relaxationFactor,args.getArgs().leftPrecond);
					    args.getMatrix());
        }

        static void deconstruct(SeqDilu<M, X, Y>* dilu)
        {
            delete dilu;
        }
#endif
    };

} // namespace Amg
} // namespace Dune
#endif // OPM_EXTRASMOOTHERS_HPP